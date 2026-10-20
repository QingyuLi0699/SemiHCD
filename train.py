"""
SemiHSCD Training Script  (v2 — minority class improvements)

Two-phase training:
  Phase 1 [0, sl_steps):  Supervised only  — FocalLoss(class_weight) + ηp*PrototypeLoss
  Phase 2 [sl_steps, total_steps): Supervised + SSL
    — FocalLoss(class_weight) + ηp*PrototypeLoss + ηs*ModifiedDINOLoss

Improvements over v1:
  1. Inverse-sqrt class weights in FocalLoss — counters majority-class dominance
  2. ClassPrototypeLoss — forces discriminative feature geometry for rare classes
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.usa  import USALoader
from data.farm import FarmLoader
from data.hermiston5 import Hermiston5Loader
from model.semi_hscd       import SemiHSCD
from model.mamba_semi_hscd import MambaSemiHSCD
from model.semi_hscd_v4    import SemiHSCDv4
from loss.focal import FocalLoss
from loss.dino  import ModifiedDINOLoss
from loss.proto import ClassPrototypeLoss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_map, gt_map, num_classes, orig_size):
    """pred_map, gt_map: numpy [H, W], values 0..K-1."""
    H, W = orig_size
    pred = pred_map[:H, :W].ravel()
    gt   = gt_map[:H, :W].ravel()

    accs = []
    for c in range(num_classes):
        mask = (gt == c)
        if mask.sum() == 0:
            accs.append(float('nan'))
        else:
            accs.append((pred[mask] == c).mean())

    OA    = (pred == gt).mean()
    AA    = float(np.nanmean(accs))

    n  = len(pred)
    po = OA
    pe = sum(((gt == c).sum() / n) * ((pred == c).sum() / n) for c in range(num_classes))
    kappa = (po - pe) / (1 - pe + 1e-8)

    return OA, AA, kappa, accs


# ---------------------------------------------------------------------------
# Class weights from full GT distribution (inverse-sqrt frequency)
# ---------------------------------------------------------------------------

def compute_class_weights(mask_tensor, num_classes, orig_size, device, c0_floor=0.0):
    """
    mask_tensor: [1, H_pad, W_pad] long
    c0_floor: minimum weight for class 0 (background). Inverse-sqrt-frequency
              weighting pushes C0's weight very low when it's the overwhelming
              majority class, which starves it of supervised gradient (doubly
              so once Focal's (1-pt)^gamma modulation compounds on top of the
              already-small per-pixel CE). A floor gives C0 more gradient
              without touching the other classes' relative weights.
    Returns [K] float tensor on device.
    """
    H, W = orig_size
    flat = mask_tensor[0, :H, :W].cpu().numpy().ravel()
    counts = np.bincount(flat, minlength=num_classes).astype(np.float32)
    w = 1.0 / np.sqrt(counts + 1.0)      # +1 avoids zero; inverse-sqrt of count
    w = w / w.mean()                      # normalize: mean weight = 1.0
    if c0_floor > 0.0:
        w[0] = max(w[0], c0_floor)
    return torch.from_numpy(w).float().to(device)


# ---------------------------------------------------------------------------
# Extract labeled pixel features from full-res feat_map
# ---------------------------------------------------------------------------

def extract_labeled_feats(feat_map, mask, train_ind, orig_size):
    """
    feat_map  : [1, D, H, W]
    mask      : [1, H, W]  long
    train_ind : [1, H, W]  float  (1 = labeled)
    Returns:
      feats  [N, D]  — raw feature vectors for labeled pixels in orig region
      labels [N]     — corresponding class labels
    """
    H, W = orig_size
    lab_mask = (train_ind[0, :H, :W] > 0)  # [H_orig, W_orig] bool
    rows, cols = lab_mask.nonzero(as_tuple=True)
    if len(rows) == 0:
        return None, None
    feats  = feat_map[0, :, rows, cols].T    # [N, D]
    labels = mask[0, rows, cols]             # [N]
    return feats, labels


# ---------------------------------------------------------------------------
# EMA teacher update
# ---------------------------------------------------------------------------

@torch.no_grad()
def ema_update(teacher, student, momentum):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.data.mul_(momentum).add_(ps.detach().data, alpha=1 - momentum)
    for bt, bs in zip(teacher.buffers(), student.buffers()):
        bt.data.copy_(bs.data)


# ---------------------------------------------------------------------------
# Learning rate schedule (cosine with linear warmup)
# ---------------------------------------------------------------------------

def get_lr(step, total_steps, base_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def adjust_momentum(step, sl_steps, total_steps, init_momentum):
    ssl_steps = total_steps - sl_steps
    if step < sl_steps:
        return init_momentum
    progress = (step - sl_steps) / max(ssl_steps, 1)
    return 1.0 + (init_momentum - 1.0) * 0.5 * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# S1: uncertainty-guided adaptive noise scale
# ---------------------------------------------------------------------------

def compute_adaptive_noise_scale(args, teacher_logit, pos, feat_hw, logit_hw,
                                  num_classes, ssl_epoch, last_avg_instability,
                                  device):
    """
    Per-pixel multiplier on the fixed noise_std, based on teacher uncertainty.
    Returns (scale [N,1] float tensor or None, stats dict).

    entropy: pixel-wise, from the teacher's CLEAN prediction this step (safe,
        no circularity — computed before any noise is added).
    temporal_instability: true pixel-wise instability needs the ALREADY-NOISED
        teacher features (this step's sim_noisy), which don't exist yet when
        we're deciding how much noise to add -> circular. Falls back to LAST
        step's global average instability, broadcast to all pixels this step.
        Logged as global, not pixel-wise (stats['pixelwise']=False).
    mixed: combines whichever of the above are available.
    """
    stats = {'mode': args.adaptive_noise_uncertainty_source, 'pixelwise': True,
              'warmup': False}

    if ssl_epoch < args.adaptive_noise_warmup_steps:
        stats['warmup'] = True
        return None, stats

    eps = 1e-6
    N = pos.shape[0]
    source = args.adaptive_noise_uncertainty_source

    entropy_norm = None
    if source in ('entropy', 'mixed'):
        Hf, Wf = feat_hw
        H_pad, W_pad = logit_hw
        ds_h, ds_w = H_pad // Hf, W_pad // Wf
        rows = (pos[:, 0] * ds_h + ds_h // 2).clamp(max=H_pad - 1)
        cols = (pos[:, 1] * ds_w + ds_w // 2).clamp(max=W_pad - 1)
        t_logit_at_pos = teacher_logit[0, :, rows, cols].T.detach()   # [N, K]
        t_prob  = F.softmax(t_logit_at_pos, dim=-1)
        entropy = -(t_prob * torch.log(t_prob + eps)).sum(dim=-1)     # [N]
        entropy_norm = (entropy / math.log(num_classes)).clamp(0.0, 1.0)
        stats['entropy_mean']      = entropy.mean().item()
        stats['entropy_min']       = entropy.min().item()
        stats['entropy_max']       = entropy.max().item()
        stats['entropy_norm_std']  = entropy_norm.std(unbiased=False).item()
        stats['entropy_norm_mean'] = entropy_norm.mean().item()

    instab_norm = None
    if source in ('temporal_instability', 'mixed'):
        instab_scalar = min(max(last_avg_instability, 0.0), 1.0)
        instab_norm = torch.full((N,), instab_scalar, device=device)
        if source == 'temporal_instability':
            stats['pixelwise'] = False
        stats['instability_mean'] = instab_scalar
        stats['instability_min']  = instab_scalar
        stats['instability_max']  = instab_scalar

    if source == 'entropy':
        uncertainty = entropy_norm
    elif source == 'temporal_instability':
        uncertainty = instab_norm
    elif source == 'mixed':
        ew, iw = args.adaptive_noise_entropy_weight, args.adaptive_noise_instability_weight
        if entropy_norm is None and instab_norm is None:
            uncertainty = None
        elif entropy_norm is None:
            print('[Warn] adaptive_noise mixed: entropy unavailable, using instability only.', flush=True)
            uncertainty = instab_norm
        elif instab_norm is None:
            print('[Warn] adaptive_noise mixed: instability unavailable, using entropy only.', flush=True)
            uncertainty = entropy_norm
        else:
            uncertainty = (ew * entropy_norm + iw * instab_norm) / (ew + iw + eps)
    else:
        raise ValueError(f'Unknown adaptive_noise_uncertainty_source: {source}')

    if uncertainty is None:
        print('[Warn] adaptive_noise: no uncertainty source available this step, skipping adaptation.', flush=True)
        return None, stats
    uncertainty = uncertainty.clamp(0.0, 1.0)

    scale = args.adaptive_noise_min_scale + \
            (args.adaptive_noise_max_scale - args.adaptive_noise_min_scale) * uncertainty
    scale = scale.detach().unsqueeze(-1)   # [N,1], detached — uncertainty carries no gradient

    stats['adaptive_scale_mean'] = scale.mean().item()
    stats['adaptive_scale_min']  = scale.min().item()
    stats['adaptive_scale_max']  = scale.max().item()
    stats['adaptive_scale_std']  = scale.std(unbiased=False).item()
    return scale, stats


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    if args.dataset == 'usa':
        loader_cls = USALoader
        data_kwargs = dict(
            t1t2_mat_path=args.t1t2_mat,
            gt_mat_path=args.gt_mat,
            split_ratio=args.split_ratio,
            np_seed=args.seed,
        )
        num_classes = 7
        orig_size   = (307, 241)
        in_channels = 154
    elif args.dataset == 'farm':
        loader_cls = FarmLoader
        data_kwargs = dict(
            t1t2_mat_path=args.t1t2_mat,
            gt_mat_path=args.gt_mat,
            split_ratio=args.split_ratio,
            np_seed=args.seed,
        )
        num_classes = 4
        orig_size   = (420, 140)
        in_channels = 154
    elif args.dataset == 'hermiston5':
        loader_cls = Hermiston5Loader
        data_kwargs = dict(
            t1t2_mat_path=args.t1t2_mat,
            gt_mat_path=args.gt_mat,
            split_ratio=args.split_ratio,
            np_seed=args.seed,
        )
        num_classes = 6
        orig_size   = (390, 200)
        in_channels = 242
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    print(f'[Info] Loading {args.dataset} dataset ...', flush=True)
    data_loader = loader_cls(**data_kwargs)
    train_loader, test_loader = data_loader.get_loaders()

    # ── Class weights (computed from full GT distribution) ───────────────────
    with torch.no_grad():
        t1_tmp, _, mask_tmp, _, _ = next(iter(train_loader))
    class_weights = compute_class_weights(mask_tmp, num_classes, orig_size, device,
                                           c0_floor=args.class0_weight_floor)
    print(f'[Info] Class weights: {class_weights.cpu().numpy().round(3)}', flush=True)

    # ── Model ───────────────────────────────────────────────────────────────
    if args.model == 'mamba':
        ModelCls = MambaSemiHSCD
    elif args.model == 'v4':
        ModelCls = SemiHSCDv4
    else:
        ModelCls = SemiHSCD
    model_kw  = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        block_channels=args.block_channels,
        num_blocks=args.num_blocks,
        reduction_ratio=args.reduction_ratio,
        inner_dim=args.inner_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_out_dim=args.proj_out_dim,
    )
    if args.model in ('mamba', 'v4'):
        model_kw['d_state'] = args.d_state
        model_kw['expand']  = args.mamba_expand
    if args.model == 'v4':
        model_kw['spec_d_inner'] = args.spec_d_inner
    print(f'[Info] Model: {ModelCls.__name__}', flush=True)

    student = ModelCls(**model_kw).to(device)
    teacher = ModelCls(**model_kw).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # ── Loss ────────────────────────────────────────────────────────────────
    criterion_sl = FocalLoss(gamma=2.0, class_weight=class_weights)

    criterion_ssl = ModifiedDINOLoss(
        out_dim=args.proj_out_dim,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.total_steps - args.sl_steps,
        psi_instability_weight=args.psi_instability_weight,
    ).to(device)

    criterion_proto = ClassPrototypeLoss(
        num_classes=num_classes,
        feat_dim=args.inner_dim,   # FPN decoder output dim
        momentum=args.proto_momentum,
    ).to(device)

    proto_class_weight = None
    if args.proto_c0_weight != 1.0:
        proto_class_weight = torch.ones(num_classes, device=device)
        proto_class_weight[0] = args.proto_c0_weight

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(criterion_ssl.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)

    ssl_started = False
    best_oa = 0.0
    best_composite = 0.0
    last_avg_instability = 0.0   # persisted across steps for temporal_instability adaptive noise
    log_lines = []

    print(f'[Info] Phase 1 (supervised): steps 0 - {args.sl_steps - 1}', flush=True)
    print(f'[Info] Phase 2 (SSL):        steps {args.sl_steps} - {args.total_steps - 1}', flush=True)

    for step in range(args.total_steps):
        student.train()

        # Fetch single full image batch
        t1, t2, mask, train_ind, unlab_ind = next(iter(train_loader))
        t1        = t1.to(device)         # [1, C, H, W]
        t2        = t2.to(device)
        mask      = mask.to(device)       # [1, H, W]
        train_ind = train_ind.to(device)  # [1, H, W]
        unlab_ind = unlab_ind.to(device)  # [1, H, W]

        # Adjust LR
        lr = get_lr(step, args.total_steps, args.lr, args.min_lr, args.warmup_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # ── Phase 1: Supervised only ─────────────────────────────────────────
        if step < args.sl_steps:
            feats_t1, feats_t2 = student.encode(t1, t2)
            logit, feat_map    = student.classify(feats_t1, feats_t2)
            loss_sl = criterion_sl(logit, mask, weight=train_ind)

            feats_lab, labels_lab = extract_labeled_feats(
                feat_map, mask, train_ind, orig_size)
            if feats_lab is not None:
                loss_proto = criterion_proto(feats_lab, labels_lab, class_weight=proto_class_weight)
            else:
                loss_proto = torch.tensor(0.0, device=device)

            loss = loss_sl + args.eta_proto * loss_proto

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % args.log_interval == 0:
                msg = (f'[Step {step+1}/{args.total_steps}] lr={lr:.5f}  '
                       f'L_sl={loss_sl.item():.4f}  L_proto={loss_proto.item():.4f}')
                print(msg, flush=True)
                log_lines.append(msg)

        # ── Phase 2: Supervised + SSL ────────────────────────────────────────
        else:
            if not ssl_started:
                teacher.load_state_dict(student.state_dict())
                ssl_started = True
                print(f'[Step {step}] Teacher initialized from student, SSL phase begins.', flush=True)

            ssl_epoch = step - args.sl_steps

            # Student forward
            out_s     = student.forward_train(t1, t2, mask, train_ind, unlab_ind)
            logit     = out_s['logit']
            feat_map  = out_s['feat_map']
            feat_t1_s = out_s['feat_t1_deep']
            feat_t2_s = out_s['feat_t2_deep']

            loss_sl = criterion_sl(logit, mask, weight=train_ind)

            # Prototype loss on labeled pixels
            feats_lab, labels_lab = extract_labeled_feats(
                feat_map, mask, train_ind, orig_size)
            if feats_lab is not None:
                loss_proto = criterion_proto(feats_lab, labels_lab, class_weight=proto_class_weight)
            else:
                loss_proto = torch.tensor(0.0, device=device)

            # SSL: ModifiedDINOLoss on unlabeled pixels
            f1_s, f2_s, diff_s, pos = student.get_ssl_pixel_feats(
                feat_t1_s, feat_t2_s, unlab_ind, args.n_ssl_pixels)

            adaptive_stats = {}
            if f1_s is not None:
                # Complementary feature-level augmentation (ssl_aug_sigma > 0):
                #   Default: student sees clean T1 + noisy T2, teacher sees noisy T1 + clean T2.
                #   With ssl_role_swap_prob > 0, the noised branch (T1 vs T2) is chosen
                #   at random each step (student/teacher always noise opposite branches,
                #   preserving the complementary property either way).
                # sigma is scaled by the clean-branch feature std for adaptive strength.
                swap = (args.ssl_aug_sigma > 0.0 and
                        torch.rand(1).item() < args.ssl_role_swap_prob)

                # Teacher's CLEAN forward is computed first (moved ahead of any
                # noise-adding on either side) so S1's entropy-guided adaptive
                # noise can read the teacher's clean per-pixel prediction
                # before deciding how much noise to add this step — computing
                # it after noise was added would be circular.
                with torch.no_grad():
                    out_t = teacher.forward_train(t1, t2, mask, train_ind, unlab_ind)
                    f1_t, f2_t, diff_t, _ = teacher.get_ssl_pixel_feats(
                        out_t['feat_t1_deep'], out_t['feat_t2_deep'],
                        unlab_ind, args.n_ssl_pixels, pos=pos)

                adaptive_scale = None
                if args.adaptive_noise_enabled and args.ssl_aug_sigma > 0.0:
                    adaptive_scale, adaptive_stats = compute_adaptive_noise_scale(
                        args, out_t['logit'], pos, feat_t1_s.shape[-2:],
                        logit.shape[-2:], num_classes, ssl_epoch,
                        last_avg_instability, device)

                s3_stats = {}
                if args.perturb_target_mode == 'fixed_complementary':
                    # ── V4-current: unchanged, exact baseline behavior ──────────
                    student_noise_abs_mean = 0.0
                    teacher_noise_abs_mean = 0.0
                    if args.ssl_aug_sigma > 0.0:
                        if not swap:
                            sigma   = (args.ssl_aug_sigma * f1_s.detach().std()).clamp(min=1e-6)
                            eff_std = sigma * adaptive_scale if adaptive_scale is not None else sigma
                            noise_s = eff_std * torch.randn_like(f2_s)
                            f2_s    = f2_s + noise_s
                        else:
                            sigma   = (args.ssl_aug_sigma * f2_s.detach().std()).clamp(min=1e-6)
                            eff_std = sigma * adaptive_scale if adaptive_scale is not None else sigma
                            noise_s = eff_std * torch.randn_like(f1_s)
                            f1_s    = f1_s + noise_s
                        diff_s = (f1_s - f2_s).abs()
                        student_noise_abs_mean = noise_s.detach().abs().mean().item()
                        if isinstance(eff_std, torch.Tensor):
                            adaptive_stats['effective_noise_std_mean'] = eff_std.mean().item()
                            adaptive_stats['effective_noise_std_min']  = eff_std.min().item()
                            adaptive_stats['effective_noise_std_max']  = eff_std.max().item()
                        else:
                            adaptive_stats['effective_noise_std_mean'] = float(eff_std)
                            adaptive_stats['effective_noise_std_min']  = float(eff_std)
                            adaptive_stats['effective_noise_std_max']  = float(eff_std)

                    s_t1   = student.proj_head.forward_pixels(f1_s)
                    s_t2   = student.proj_head.forward_pixels(f2_s)
                    s_diff = student.proj_head.forward_pixels(diff_s)

                    with torch.no_grad():
                        t_t1_psi = None
                        t_t2_psi = None
                        if args.ssl_aug_sigma > 0.0:
                            if not swap:
                                t_t1_psi = teacher.proj_head.forward_pixels(f1_t)   # clean T1, for psi
                                eff_std_t = sigma * adaptive_scale if adaptive_scale is not None else sigma
                                noise_t   = eff_std_t * torch.randn_like(f1_t)
                                f1_t      = f1_t + noise_t                          # noisy
                            else:
                                t_t2_psi = teacher.proj_head.forward_pixels(f2_t)   # clean T2, for psi
                                eff_std_t = sigma * adaptive_scale if adaptive_scale is not None else sigma
                                noise_t   = eff_std_t * torch.randn_like(f2_t)
                                f2_t      = f2_t + noise_t                          # noisy
                            diff_t = (f1_t - f2_t).abs()
                            teacher_noise_abs_mean = noise_t.detach().abs().mean().item()
                        t_t1   = teacher.proj_head.forward_pixels(f1_t)
                        t_t2   = teacher.proj_head.forward_pixels(f2_t)
                        t_diff = teacher.proj_head.forward_pixels(diff_t)

                    adaptive_stats['student_noise_abs_mean'] = student_noise_abs_mean
                    adaptive_stats['teacher_noise_abs_mean'] = teacher_noise_abs_mean

                else:
                    # ── S3: psi-guided perturbation target ──────────────────────
                    # psi computed here (same formula/inputs as inside criterion_ssl,
                    # using the still-100%-clean f1_t/f2_t) so it can decide, BEFORE
                    # any noise is added, how much of this pixel's noise budget goes
                    # to the temporal branch vs. the diff branch. Recomputed again
                    # inside criterion_ssl for the actual loss weighting (same
                    # formula + same clean inputs -> numerically identical; not a
                    # second, different psi).
                    with torch.no_grad():
                        t_t1_psi = teacher.proj_head.forward_pixels(f1_t)
                        t_t2_psi = teacher.proj_head.forward_pixels(f2_t)
                        sim_perturb = F.cosine_similarity(
                            t_t1_psi - criterion_ssl.center,
                            t_t2_psi - criterion_ssl.center, dim=-1)
                        psi_raw = ((1.0 + sim_perturb) / 2.0) ** criterion_ssl.gamma   # [N]
                        psi_for_perturb = psi_raw.detach() if args.psi_detach_for_perturb else psi_raw
                        psi_col = psi_for_perturb.unsqueeze(-1)   # [N,1], broadcasts vs [N,C]

                        if args.perturb_target_mode == 'psi_guided_hard':
                            temporal_w = (psi_col >= args.psi_hard_thr).float()
                            diff_w     = 1.0 - temporal_w
                        else:  # psi_guided_soft
                            temporal_w = psi_col
                            diff_w     = 1.0 - psi_col

                    temporal_std_cfg = (args.temporal_noise_std if args.temporal_noise_std is not None
                                        else args.ssl_aug_sigma)
                    diff_std_cfg = (args.diff_noise_std if args.diff_noise_std is not None
                                    else args.ssl_aug_sigma)
                    noise_scale = adaptive_scale if adaptive_scale is not None else 1.0

                    # ---- Student (grad-tracked) ----
                    diff_s_clean = (f1_s - f2_s).abs()
                    if not swap:
                        sigma_temp_s = (temporal_std_cfg * f2_s.detach().std()).clamp(min=1e-6)
                        noise_temporal_s = sigma_temp_s * noise_scale * torch.randn_like(f2_s)
                        f2_s_pert   = f2_s + temporal_w * noise_temporal_s
                        diff_s_temp = (f1_s - f2_s_pert).abs()
                        s_t1_in, s_t2_in = f1_s, f2_s_pert
                    else:
                        sigma_temp_s = (temporal_std_cfg * f1_s.detach().std()).clamp(min=1e-6)
                        noise_temporal_s = sigma_temp_s * noise_scale * torch.randn_like(f1_s)
                        f1_s_pert   = f1_s + temporal_w * noise_temporal_s
                        diff_s_temp = (f1_s_pert - f2_s).abs()
                        s_t1_in, s_t2_in = f1_s_pert, f2_s

                    if args.diff_noise_enabled:
                        sigma_diff_s = (diff_std_cfg * diff_s_clean.detach().std()).clamp(min=1e-6)
                        noise_diff_s = sigma_diff_s * noise_scale * torch.randn_like(diff_s_clean)
                    else:
                        noise_diff_s = torch.zeros_like(diff_s_clean)
                    diff_s_noisy = diff_s_clean + diff_w * noise_diff_s

                    s_t1   = student.proj_head.forward_pixels(s_t1_in)
                    s_t2   = student.proj_head.forward_pixels(s_t2_in)
                    s_diff = student.proj_head.forward_pixels(
                        temporal_w * diff_s_temp + diff_w * diff_s_noisy)

                    # ---- Teacher (no_grad) ----
                    with torch.no_grad():
                        diff_t_clean = (f1_t - f2_t).abs()
                        if not swap:
                            sigma_temp_t = (temporal_std_cfg * f1_t.detach().std()).clamp(min=1e-6)
                            noise_temporal_t = sigma_temp_t * noise_scale * torch.randn_like(f1_t)
                            f1_t_pert   = f1_t + temporal_w * noise_temporal_t
                            diff_t_temp = (f1_t_pert - f2_t).abs()
                            t_t1_in, t_t2_in = f1_t_pert, f2_t
                        else:
                            sigma_temp_t = (temporal_std_cfg * f2_t.detach().std()).clamp(min=1e-6)
                            noise_temporal_t = sigma_temp_t * noise_scale * torch.randn_like(f2_t)
                            f2_t_pert   = f2_t + temporal_w * noise_temporal_t
                            diff_t_temp = (f1_t - f2_t_pert).abs()
                            t_t1_in, t_t2_in = f1_t, f2_t_pert

                        if args.diff_noise_enabled:
                            sigma_diff_t = (diff_std_cfg * diff_t_clean.detach().std()).clamp(min=1e-6)
                            noise_diff_t = sigma_diff_t * noise_scale * torch.randn_like(diff_t_clean)
                        else:
                            noise_diff_t = torch.zeros_like(diff_t_clean)
                        diff_t_noisy = diff_t_clean + diff_w * noise_diff_t

                        t_t1   = teacher.proj_head.forward_pixels(t_t1_in)
                        t_t2   = teacher.proj_head.forward_pixels(t_t2_in)
                        t_diff = teacher.proj_head.forward_pixels(
                            temporal_w * diff_t_temp + diff_w * diff_t_noisy)

                    s3_stats = {
                        'temporal_noise_weight_mean': temporal_w.mean().item(),
                        'diff_noise_weight_mean':     diff_w.mean().item(),
                        'temporal_noise_abs_mean_student': noise_temporal_s.detach().abs().mean().item(),
                        'temporal_noise_abs_mean_teacher': noise_temporal_t.detach().abs().mean().item(),
                        'diff_noise_abs_mean_student':     noise_diff_s.detach().abs().mean().item(),
                        'diff_noise_abs_mean_teacher':     noise_diff_t.detach().abs().mean().item(),
                        'effective_temporal_noise_abs_mean': 0.5 * (
                            (temporal_w * noise_temporal_s).detach().abs().mean().item() +
                            (temporal_w * noise_temporal_t).detach().abs().mean().item()),
                        'effective_diff_noise_abs_mean': 0.5 * (
                            (diff_w * noise_diff_s).detach().abs().mean().item() +
                            (diff_w * noise_diff_t).detach().abs().mean().item()),
                    }
                    if args.perturb_target_mode == 'psi_guided_hard':
                        s3_stats['unchanged_like_ratio'] = (psi_col >= args.psi_hard_thr).float().mean().item()
                        s3_stats['changed_like_ratio']   = (psi_col < args.psi_hard_thr).float().mean().item()

                loss_ssl, dino_stats = criterion_ssl(s_t1, s_t2, s_diff,
                                         t_t1, t_t2, t_diff, ssl_epoch,
                                         t_t1_psi=t_t1_psi, t_t2_psi=t_t2_psi,
                                         return_stats=True)

                # Clean-vs-noised instability for this step, computed once and
                # reused by (a) the C0 trust signal below and (b) the teacher
                # EMA momentum modulation after the optimizer step: if this
                # step's augmentation flipped the teacher's unchanged/changed
                # read a lot, the gradient signal this step is less trustworthy,
                # so the teacher should update more conservatively (see
                # --teacher_momentum_instability_scale).
                avg_instability = 0.0
                sim_clean = sim_noisy = None
                if args.ssl_aug_sigma > 0.0:
                    with torch.no_grad():
                        clean_t1 = t_t1_psi if t_t1_psi is not None else t_t1
                        clean_t2 = t_t2_psi if t_t2_psi is not None else t_t2
                        center = criterion_ssl.center
                        sim_clean = F.cosine_similarity(
                            clean_t1 - center, clean_t2 - center, dim=-1)
                        sim_noisy = F.cosine_similarity(
                            t_t1 - center, t_t2 - center, dim=-1)
                        avg_instability = (sim_clean - sim_noisy).abs().mean().item()
                last_avg_instability = avg_instability   # for next step's temporal_instability mode

                # Noise-invariance C0 trust signal: pixels whose teacher T1/T2
                # similarity says "unchanged" both before AND after the noise
                # that was actually applied this step are high-confidence
                # background — feed them to the classifier as free C0
                # pseudo-labels at their full-res coordinates.
                # Delayed + ramped: right after teacher init (ssl_epoch=0) its
                # similarity judgments are uncalibrated, so trusting them at
                # full strength immediately injects wrong pseudo-labels (this
                # is what caused the step-600 collapse in the first attempt).
                if ssl_epoch < args.c0_trust_warmup_steps:
                    eff_eta_c0_trust = 0.0
                else:
                    ramp = min(1.0, (ssl_epoch - args.c0_trust_warmup_steps) /
                               max(args.c0_trust_ramp_steps, 1))
                    eff_eta_c0_trust = args.eta_c0_trust * ramp

                loss_c0_trust = torch.tensor(0.0, device=device)
                if eff_eta_c0_trust > 0.0 and sim_clean is not None:
                    with torch.no_grad():
                        trust_mask = (sim_clean > args.c0_trust_tau) & \
                                     (sim_noisy > args.c0_trust_tau)

                    if trust_mask.any():
                        Hf, Wf   = feat_t1_s.shape[-2:]
                        H_pad, W_pad = logit.shape[-2:]
                        ds_h, ds_w = H_pad // Hf, W_pad // Wf
                        rows = (pos[trust_mask, 0] * ds_h + ds_h // 2).clamp(max=H_pad - 1)
                        cols = (pos[trust_mask, 1] * ds_w + ds_w // 2).clamp(max=W_pad - 1)
                        logit_trust = logit[0, :, rows, cols].T          # [M, K]
                        pseudo_c0   = torch.zeros(logit_trust.size(0),
                                                   dtype=torch.long, device=device)
                        loss_c0_trust = F.cross_entropy(logit_trust, pseudo_c0)

                loss = (loss_sl + args.eta_proto * loss_proto +
                        args.eta * loss_ssl + eff_eta_c0_trust * loss_c0_trust)
            else:
                loss     = loss_sl + args.eta_proto * loss_proto
                loss_ssl = torch.tensor(0.0)
                avg_instability = 0.0
                dino_stats = {}
                s3_stats   = {}

            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
            optimizer.step()

            # EMA update
            momentum = adjust_momentum(step, args.sl_steps, args.total_steps,
                                       args.teacher_momentum)
            if args.teacher_momentum_instability_scale > 0.0:
                # This step's augmentation flipped the teacher's unchanged/
                # changed read a lot -> this step's signal is less trustworthy
                # -> update the teacher more conservatively (push momentum
                # toward 1, i.e. slower update) instead of the fixed schedule.
                momentum = momentum + (1.0 - momentum) * \
                           args.teacher_momentum_instability_scale * avg_instability
                momentum = min(momentum, 0.9999)
            ema_update(teacher, student, momentum)

            if (step + 1) % args.log_interval == 0:
                msg = (f'[Step {step+1}/{args.total_steps}] lr={lr:.5f}  '
                       f'L_sl={loss_sl.item():.4f}  '
                       f'L_proto={loss_proto.item():.4f}  '
                       f'L_ssl={loss_ssl.item():.4f}')
                print(msg, flush=True)
                log_lines.append(msg)

                if args.adaptive_noise_enabled:
                    s = adaptive_stats
                    adapt_msg = (
                        f'  [AdaptiveNoise] enabled={args.adaptive_noise_enabled} '
                        f'mode={s.get("mode", args.adaptive_noise_uncertainty_source)} '
                        f'pixelwise={s.get("pixelwise", "n/a")} warmup={s.get("warmup", False)} '
                        f'base_std_cfg={args.ssl_aug_sigma} '
                        f'min_scale={args.adaptive_noise_min_scale} max_scale={args.adaptive_noise_max_scale}\n'
                        f'  entropy: mean={s.get("entropy_mean", float("nan")):.4f} '
                        f'min={s.get("entropy_min", float("nan")):.4f} '
                        f'max={s.get("entropy_max", float("nan")):.4f} '
                        f'norm_mean={s.get("entropy_norm_mean", float("nan")):.4f}\n'
                        f'  instability: mean={s.get("instability_mean", float("nan")):.4f} '
                        f'min={s.get("instability_min", float("nan")):.4f} '
                        f'max={s.get("instability_max", float("nan")):.4f}\n'
                        f'  adaptive_scale: mean={s.get("adaptive_scale_mean", float("nan")):.4f} '
                        f'min={s.get("adaptive_scale_min", float("nan")):.4f} '
                        f'max={s.get("adaptive_scale_max", float("nan")):.4f}\n'
                        f'  effective_noise_std: mean={s.get("effective_noise_std_mean", float("nan")):.6f} '
                        f'min={s.get("effective_noise_std_min", float("nan")):.6f} '
                        f'max={s.get("effective_noise_std_max", float("nan")):.6f}\n'
                        f'  noise_abs_mean: student={s.get("student_noise_abs_mean", float("nan")):.6f} '
                        f'teacher={s.get("teacher_noise_abs_mean", float("nan")):.6f}'
                    )
                    print(adapt_msg, flush=True)
                    log_lines.append(adapt_msg)

                ds = dino_stats
                psi_msg = (
                    f'  [Psi] perturb_target_mode={args.perturb_target_mode} '
                    f'psi_mean={ds.get("psi_mean", float("nan")):.4f} '
                    f'psi_min={ds.get("psi_min", float("nan")):.4f} '
                    f'psi_max={ds.get("psi_max", float("nan")):.4f} '
                    f'psi_std={ds.get("psi_std", float("nan")):.4f}  '
                    f'loss_u={ds.get("loss_u", float("nan")):.4f} '
                    f'loss_d={ds.get("loss_d", float("nan")):.4f}'
                )
                print(psi_msg, flush=True)
                log_lines.append(psi_msg)

                if args.perturb_target_mode != 'fixed_complementary':
                    s = s3_stats
                    s3_msg = (
                        f'  [S3] temporal_noise_weight_mean={s.get("temporal_noise_weight_mean", float("nan")):.4f} '
                        f'diff_noise_weight_mean={s.get("diff_noise_weight_mean", float("nan")):.4f}\n'
                        f'  [S3] temporal_noise_abs_mean: student={s.get("temporal_noise_abs_mean_student", float("nan")):.6f} '
                        f'teacher={s.get("temporal_noise_abs_mean_teacher", float("nan")):.6f}\n'
                        f'  [S3] diff_noise_abs_mean: student={s.get("diff_noise_abs_mean_student", float("nan")):.6f} '
                        f'teacher={s.get("diff_noise_abs_mean_teacher", float("nan")):.6f}\n'
                        f'  [S3] effective_temporal_noise_abs_mean={s.get("effective_temporal_noise_abs_mean", float("nan")):.6f} '
                        f'effective_diff_noise_abs_mean={s.get("effective_diff_noise_abs_mean", float("nan")):.6f}'
                    )
                    if args.perturb_target_mode == 'psi_guided_hard':
                        s3_msg += (f'\n  [S3] unchanged_like_ratio={s.get("unchanged_like_ratio", float("nan")):.4f} '
                                   f'changed_like_ratio={s.get("changed_like_ratio", float("nan")):.4f}')
                    print(s3_msg, flush=True)
                    log_lines.append(s3_msg)

        # ── Evaluation ───────────────────────────────────────────────────────
        if (step + 1) % args.eval_interval == 0 or step == args.total_steps - 1:
            student.eval()
            with torch.no_grad():
                t1e, t2e, maske, test_ind, _ = next(iter(test_loader))
                t1e, t2e = t1e.to(device), t2e.to(device)
                prob = student(t1e, t2e)                   # [1, K, H, W]
                pred = prob[0].argmax(dim=0).cpu().numpy() # [H, W]
                gt   = maske[0].numpy()                    # [H, W]

            OA, AA, kappa, accs = compute_metrics(pred, gt, num_classes, orig_size)
            composite = (OA + AA + kappa) / 3.0
            rare_accs = accs[1:]   # C1..C_{K-1}, excludes C0/background
            rare_mean = float(np.nanmean(rare_accs)) if len(rare_accs) > 0 else float('nan')
            acc_str = '  '.join(f'C{i}={a:.4f}' for i, a in enumerate(accs))
            msg = (f'[Eval {step+1}] OA={OA:.4f}  AA={AA:.4f}  Kappa={kappa:.4f}  '
                   f'Composite={composite:.4f}  C1-C{num_classes-1}Mean={rare_mean:.4f}\n'
                   f'         {acc_str}')
            print(msg, flush=True)
            log_lines.append(msg)
            # Write progress file so step is visible even while buffered
            with open(os.path.join(args.log_dir, 'progress.txt'), 'w') as _pf:
                _pf.write(msg + '\n')

            if OA > best_oa:
                best_oa = OA
                torch.save(student.state_dict(),
                           os.path.join(args.log_dir, 'best_model.pth'))
            if composite > best_composite:
                best_composite = composite
                torch.save(student.state_dict(),
                           os.path.join(args.log_dir, 'best_model_composite.pth'))

    # ── Final log ────────────────────────────────────────────────────────────
    with open(os.path.join(args.log_dir, 'train.log'), 'w') as f:
        f.write('\n'.join(log_lines))

    print(f'\n[Done] Best OA: {best_oa:.4f}  Best Composite: {best_composite:.4f}  '
          f'Log: {args.log_dir}', flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',   default='usa', choices=['usa', 'farm', 'hermiston5'])
    p.add_argument('--t1t2_mat',  required=True)
    p.add_argument('--gt_mat',    required=True)
    p.add_argument('--log_dir',   default='./logs/semi_hscd')
    p.add_argument('--gpu',       type=int, default=0)
    p.add_argument('--seed',      type=int, default=2333)
    p.add_argument('--split_ratio', type=float, default=0.01)

    # Model selection
    p.add_argument('--model', choices=['cnn', 'mamba', 'v4'], default='cnn',
                   help='cnn=SemiHSCD (v1/v2), mamba=MambaSemiHSCD (v3), v4=SemiHSCDv4')

    # Architecture
    p.add_argument('--block_channels',   type=int, nargs=4, default=[96, 128, 192, 256])
    p.add_argument('--num_blocks',       type=int, nargs=4, default=[1, 1, 1, 1])
    p.add_argument('--reduction_ratio',  type=float, default=1.0)
    p.add_argument('--inner_dim',        type=int, default=128)
    p.add_argument('--proj_hidden_dim',  type=int, default=256)
    p.add_argument('--proj_out_dim',     type=int, default=512)

    # Mamba-specific
    p.add_argument('--d_state',      type=int,   default=16,
                   help='SSM state dimension (Mamba N)')
    p.add_argument('--mamba_expand', type=int,   default=1,
                   help='SS2D channel expansion factor')
    p.add_argument('--spec_d_inner', type=int,   default=32,
                   help='(v4) SpectralSSM1D inner dim')

    # Training
    p.add_argument('--total_steps', type=int, default=1000)
    p.add_argument('--sl_steps',    type=int, default=300)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--min_lr',      type=float, default=1e-5)
    p.add_argument('--warmup_steps', type=int,  default=30)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--clip_grad',    type=float, default=3.0)
    p.add_argument('--class0_weight_floor', type=float, default=0.0,
                   help='Minimum FocalLoss class_weight for class 0 (background). '
                        'Inverse-sqrt-frequency weighting pushes C0 very low when it '
                        'dominates pixel count, starving it of supervised gradient. '
                        '0.0 = disabled (default, original inverse-sqrt-freq weight).')

    # SSL
    p.add_argument('--eta',                      type=float, default=0.5)
    p.add_argument('--ssl_aug_sigma',            type=float, default=0.0,
                   help='Feature-level Gaussian noise for complementary augmentation '
                        '(student T2 noisy / teacher T1 noisy). 0=disabled (default).')
    p.add_argument('--ssl_role_swap_prob',       type=float, default=0.0,
                   help='Probability of swapping which temporal branch (T1 vs T2) gets '
                        'noised each step, keeping the complementary property (student '
                        'and teacher always noise opposite branches). 0.0 = fixed '
                        'assignment (student noises T2, teacher noises T1), default.')
    p.add_argument('--c0_trust_tau',             type=float, default=0.8,
                   help='Cosine-similarity threshold on teacher T1/T2 agreement (clean '
                        'AND noised) above which an unlabeled pixel is treated as a '
                        'high-confidence background (C0) pseudo-label.')
    p.add_argument('--eta_c0_trust',             type=float, default=0.0,
                   help='Target weight for the extra CE loss on noise-invariant '
                        'high-confidence C0 pseudo-labels. 0=disabled (default).')
    p.add_argument('--c0_trust_warmup_steps',    type=int,   default=0,
                   help='SSL-relative steps (from teacher init) before the C0 trust '
                        'signal activates at all, giving the teacher time to calibrate '
                        'before its similarity judgments are trusted for pseudo-labels.')
    p.add_argument('--c0_trust_ramp_steps',      type=int,   default=1,
                   help='SSL-relative steps over which eta_c0_trust ramps linearly from '
                        '0 to its target value after warmup ends. 1 = no ramp (default).')
    p.add_argument('--psi_instability_weight',   type=float, default=0.0,
                   help='Soften psi toward 0.5 (hedge L_u/L_d) when clean-vs-noised '
                        'similarity disagree a lot for a pixel (unstable unchanged/'
                        'changed judgment under the augmentation actually applied this '
                        'step). 0.0 = disabled (default, original psi formula).')
    p.add_argument('--teacher_momentum_instability_scale', type=float, default=0.0,
                   help='Push teacher EMA momentum toward 1 (slower update) on steps '
                        'where clean-vs-noised similarity disagree a lot on average '
                        '(this step\'s SSL signal is less trustworthy). Reuses the same '
                        'instability signal as psi_instability_weight, applied to the '
                        'teacher update rate instead of psi. 0.0 = disabled (default, '
                        'fixed cosine momentum schedule).')
    # S1: uncertainty-guided adaptive temporal perturbation. Disabled by
    # default -> exactly reproduces the fixed-noise_std baseline.
    p.add_argument('--adaptive_noise_enabled', action='store_true',
                   help='Scale the fixed feature-level noise_std per-pixel by teacher '
                        'uncertainty instead of using a single fixed value for the whole '
                        'batch. Off by default (fully reproduces current fixed-noise behavior).')
    p.add_argument('--adaptive_noise_uncertainty_source', type=str, default='entropy',
                   choices=['entropy', 'temporal_instability', 'mixed'],
                   help='entropy: pixel-wise, from teacher clean prediction entropy this '
                        'step (no circularity). temporal_instability: true pixel-wise value '
                        'needs already-noised features, which do not exist yet when deciding '
                        'noise strength -> falls back to LAST step\'s global average '
                        'instability broadcast to all pixels (logged as global, not '
                        'pixel-wise). mixed: weighted combination of both when available.')
    p.add_argument('--adaptive_noise_min_scale', type=float, default=0.5,
                   help='Noise scale multiplier when teacher is maximally confident (low uncertainty).')
    p.add_argument('--adaptive_noise_max_scale', type=float, default=2.0,
                   help='Noise scale multiplier when teacher is maximally uncertain.')
    p.add_argument('--adaptive_noise_entropy_weight', type=float, default=1.0,
                   help='Weight for the entropy term in mixed mode.')
    p.add_argument('--adaptive_noise_instability_weight', type=float, default=1.0,
                   help='Weight for the temporal-instability term in mixed mode.')
    p.add_argument('--adaptive_noise_warmup_steps', type=int, default=0,
                   help='SSL-relative steps before adaptive noise activates; before that, '
                        'falls back to the fixed noise_std (teacher is freshly initialized '
                        'and its uncertainty estimate is not yet calibrated).')

    # ── S3: psi-guided perturbation target ──────────────────────────────
    p.add_argument('--perturb_target_mode', type=str, default='fixed_complementary',
                   choices=['fixed_complementary', 'psi_guided_soft', 'psi_guided_hard'],
                   help='fixed_complementary: current V4 behavior (student always noises '
                        'T2, teacher always noises T1, exactly reproduces baseline). '
                        'psi_guided_soft: psi (same clean-teacher-projection formula used '
                        'for loss weighting) also continuously blends how much of the noise '
                        'budget goes to the temporal branch vs. the diff branch per pixel. '
                        'psi_guided_hard: same idea with a hard threshold instead of a '
                        'continuous blend.')
    p.add_argument('--psi_detach_for_perturb', action=argparse.BooleanOptionalAction, default=True,
                   help='Detach psi before using it to weight perturbation targets, so the '
                        'noise-injection path cannot backprop into psi. Default True.')
    p.add_argument('--psi_guided_eps',   type=float, default=1e-6)
    p.add_argument('--psi_hard_thr',     type=float, default=0.5,
                   help='psi_guided_hard only: psi >= thr -> temporal-perturb this pixel, '
                        'psi < thr -> diff-perturb this pixel.')
    p.add_argument('--temporal_noise_std', type=float, default=None,
                   help='S3 temporal-branch noise std multiplier (same convention as '
                        'ssl_aug_sigma: multiplied by the clean branch feature std). '
                        'Defaults to ssl_aug_sigma if not set.')
    p.add_argument('--diff_noise_std',   type=float, default=None,
                   help='S3 diff-branch noise std multiplier. Defaults to ssl_aug_sigma if not set.')
    p.add_argument('--diff_noise_enabled', action=argparse.BooleanOptionalAction, default=True,
                   help='S3 only: whether the diff branch gets its own independent noise '
                        'term at all (psi_guided modes still always compute a clean-diff '
                        'baseline regardless of this flag).')

    p.add_argument('--n_ssl_pixels',             type=int,   default=2048)
    p.add_argument('--teacher_momentum',         type=float, default=0.996)
    p.add_argument('--warmup_teacher_temp',      type=float, default=0.04)
    p.add_argument('--teacher_temp',             type=float, default=0.07)
    p.add_argument('--warmup_teacher_temp_epochs', type=int, default=10)

    # Prototype loss
    p.add_argument('--eta_proto',       type=float, default=0.1,
                   help='Weight for prototype contrastive loss')
    p.add_argument('--proto_momentum',  type=float, default=0.9,
                   help='EMA momentum for prototype update (lower = faster adapt for rare classes)')
    p.add_argument('--proto_c0_weight', type=float, default=1.0,
                   help='Extra multiplier on class-0 (background) pixels within loss_proto, '
                        'other classes unchanged. 1.0 = plain mean (default behavior).')

    # Logging
    p.add_argument('--log_interval',  type=int, default=100)
    p.add_argument('--eval_interval', type=int, default=100)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
