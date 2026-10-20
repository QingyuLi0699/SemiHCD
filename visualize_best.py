import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from PIL import Image

from data.farm import FarmLoader
from data.hermiston5 import Hermiston5Loader
from data.usa import USALoader
from model.semi_hscd import SemiHSCD
from model.mamba_semi_hscd import MambaSemiHSCD
from model.semi_hscd_v4 import SemiHSCDv4
from train import compute_metrics


PALETTES = {
    "farm": np.array([
        [0, 0, 0],
        [216, 56, 58],
        [0, 0, 255],
        [0, 255, 0],
    ], dtype=np.uint8),
    "usa": np.array([
        [238, 238, 238],
        [216, 56, 58],
        [44, 135, 196],
        [71, 173, 97],
        [246, 178, 62],
        [142, 91, 178],
        [0, 153, 153],
    ], dtype=np.uint8),
    "hermiston5": np.array([
        [238, 238, 238],
        [216, 56, 58],
        [44, 135, 196],
        [71, 173, 97],
        [246, 178, 62],
        [142, 91, 178],
    ], dtype=np.uint8),
}


def build_loader(args):
    if args.dataset == "farm":
        return FarmLoader(args.t1t2_mat, args.gt_mat, args.split_ratio, args.seed), 4
    if args.dataset == "usa":
        return USALoader(args.t1t2_mat, args.gt_mat, args.split_ratio, args.seed), 7
    if args.dataset == "hermiston5":
        return Hermiston5Loader(args.t1t2_mat, args.gt_mat, args.split_ratio, args.seed), 6
    raise ValueError(args.dataset)


def build_model(args, in_channels, num_classes):
    model_kw = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        block_channels=args.block_channels,
        num_blocks=args.num_blocks,
        reduction_ratio=args.reduction_ratio,
        inner_dim=args.inner_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_out_dim=args.proj_out_dim,
    )
    if args.model in ("mamba", "v4"):
        model_kw["d_state"] = args.d_state
        model_kw["expand"] = args.mamba_expand
    if args.model == "v4":
        model_kw["spec_d_inner"] = args.spec_d_inner

    if args.model == "cnn":
        return SemiHSCD(**model_kw)
    if args.model == "mamba":
        return MambaSemiHSCD(**model_kw)
    if args.model == "v4":
        return SemiHSCDv4(**model_kw)
    raise ValueError(args.model)


def colorize(label_map, dataset):
    palette = PALETTES[dataset]
    label = np.asarray(label_map, dtype=np.int64)
    label = np.clip(label, 0, len(palette) - 1)
    return palette[label]


def save_panel(pred_rgb, gt_rgb, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    axes[0].imshow(pred_rgb)
    axes[0].set_title("Prediction")
    axes[0].axis("off")
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["farm", "usa", "hermiston5"], required=True)
    p.add_argument("--t1t2_mat", required=True)
    p.add_argument("--gt_mat", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2333)
    p.add_argument("--split_ratio", type=float, default=0.20)
    p.add_argument("--model", choices=["cnn", "mamba", "v4"], default="v4")
    p.add_argument("--block_channels", type=int, nargs=4, default=[96, 128, 192, 256])
    p.add_argument("--num_blocks", type=int, nargs=4, default=[1, 2, 2, 2])
    p.add_argument("--reduction_ratio", type=float, default=1.0)
    p.add_argument("--inner_dim", type=int, default=128)
    p.add_argument("--proj_hidden_dim", type=int, default=256)
    p.add_argument("--proj_out_dim", type=int, default=512)
    p.add_argument("--d_state", type=int, default=8)
    p.add_argument("--mamba_expand", type=int, default=1)
    p.add_argument("--spec_d_inner", type=int, default=32)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    loader_obj, num_classes = build_loader(args)
    _, test_loader = loader_obj.get_loaders()
    t1, t2, mask, _, _ = next(iter(test_loader))
    in_channels = t1.shape[1]
    orig_size = test_loader.dataset.orig_size

    model = build_model(args, in_channels, num_classes).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        prob = model(t1.to(device), t2.to(device))
        pred_pad = prob[0].argmax(dim=0).cpu().numpy().astype(np.uint8)

    h, w = orig_size
    pred = pred_pad[:h, :w]
    gt = mask[0, :h, :w].numpy().astype(np.uint8)

    oa, aa, kappa, accs = compute_metrics(pred, gt, num_classes, orig_size)
    pred_rgb = colorize(pred, args.dataset)
    gt_rgb = colorize(gt, args.dataset)

    stem = f"{args.dataset}_seed{args.seed}"
    Image.fromarray(pred_rgb).save(os.path.join(args.out_dir, f"{stem}_pred_color.png"))
    Image.fromarray(gt_rgb).save(os.path.join(args.out_dir, f"{stem}_gt_color.png"))
    Image.fromarray(pred).save(os.path.join(args.out_dir, f"{stem}_pred_label.png"))
    Image.fromarray(gt).save(os.path.join(args.out_dir, f"{stem}_gt_label.png"))
    save_panel(
        pred_rgb, gt_rgb,
        os.path.join(args.out_dir, f"{stem}_pred_vs_gt.png"),
        f"{args.dataset} seed {args.seed} OA={oa:.4f} AA={aa:.4f} Kappa={kappa:.4f}",
    )
    sio.savemat(os.path.join(args.out_dir, f"{stem}_prediction.mat"), {
        "pred": pred,
        "gt": gt,
        "prob": prob[0, :, :h, :w].cpu().numpy(),
        "OA": oa,
        "AA": aa,
        "Kappa": kappa,
        "class_acc": np.asarray(accs),
    })

    acc_str = " ".join(f"C{i}={a:.4f}" for i, a in enumerate(accs))
    metrics_line = (
        f"{args.dataset} seed={args.seed} checkpoint={args.checkpoint}\n"
        f"OA={oa:.4f} AA={aa:.4f} Kappa={kappa:.4f}\n"
        f"{acc_str}\n"
    )
    with open(os.path.join(args.out_dir, f"{stem}_metrics.txt"), "w") as f:
        f.write(metrics_line)
    print(metrics_line, flush=True)
    print(f"Saved visualization to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
