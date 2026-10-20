# SemiHCD: SemiHSCDv4 S13-A1

This repository contains the **S13-A1 semi-supervised hyperspectral change-detection training path** from the local SemiHSCD project. It combines the V4 CNN-Mamba network, supervised Focal and prototype losses, teacher-student feature consistency, entropy-guided feature-noise strength (S1, scale 0.5-2.0), and psi-guided soft allocation of temporal/difference perturbations (S3). This repository contains source code only; it does not contain datasets, checkpoints, or experiment logs.

## Code layout

- `train.py`: two-phase training and evaluation; selects `--model v4` in the S13-A1 launchers.
- `model/semi_hscd_v4.py`, `model/conv_mamba.py`, `model/mamba_encoder.py`: V4 network and custom selective scan.
- `model/semi_hscd.py`, `model/mamba_semi_hscd.py`: shared building blocks and older model classes imported by `train.py`.
- `loss/`: Focal, labeled-pixel prototype, and teacher-student consistency losses.
- `data/`: full-image loaders, per-band normalization, stratified pixel split, padding.
- `run_v4_s13a1_{farm20,usa20,hermiston5}.sh`: original 20%-label S13-A1 commands.
- `visualize_best.py`: optional checkpoint inference and visualization.

## Environment

Use Python with PyTorch and install the dependencies in `requirements.txt`. The data loader also imports `simplecv.data.preprocess.divisible_pad` from [Z-Zheng/SimpleCV](https://github.com/Z-Zheng/SimpleCV). Install a compatible SimpleCV checkout separately; a fresh-environment dependency lock has **not** been verified. The local experiments used `/root/anaconda3/envs/MambaHSI_env/bin/python`.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps git+https://github.com/Z-Zheng/SimpleCV.git
```

The supplied launchers retain the original machine's absolute Python and `.mat` paths for provenance. Change those paths for another machine, or invoke `train.py` directly. Each image MAT must contain `T1` and `T2` arrays of shape `[H,W,C]`; the GT MAT must contain `Multitype` of shape `[H,W]` with integer labels `0..K-1`. Expected configurations are:

| Dataset | Image size | Bands | Classes | `d_state` | Class-0 floor |
|---|---:|---:|---:|---:|---:|
| `farm` | 420 x 140 | 154 | 4 | 16 | 0.5 |
| `usa` | 307 x 241 | 154 | 7 | 8 | 0.3 |
| `hermiston5` | 390 x 200 | 242 | 6 | 8 | 0.3 |

For example, after setting `IMAGE_MAT` and `GT_MAT` to the Farm files:

```bash
python train.py --dataset farm --t1t2_mat "$IMAGE_MAT" --gt_mat "$GT_MAT" \
  --split_ratio 0.20 --log_dir ./logs/s13a1_farm_seed2333 \
  --model v4 --gpu 0 --seed 2333 --d_state 16 --mamba_expand 1 \
  --spec_d_inner 32 --num_blocks 1 2 2 2 --total_steps 2000 --sl_steps 400 \
  --log_interval 200 --eval_interval 200 --eta 0.25 --eta_proto 0.3 \
  --proto_momentum 0.9 --ssl_aug_sigma 0.1 --class0_weight_floor 0.5 \
  --perturb_target_mode psi_guided_soft --adaptive_noise_enabled \
  --adaptive_noise_uncertainty_source entropy \
  --adaptive_noise_min_scale 0.5 --adaptive_noise_max_scale 2.0
```

For USA and Hermiston5, change `--dataset`, paths, `--d_state 8`, and `--class0_weight_floor 0.3`. Run seeds 2333, 2334, and 2335 separately for three-seed results. The first 400 iterations use labeled-pixel Focal and prototype losses; subsequent iterations also use sampled unlabeled feature consistency. The Teacher is copied from the Student at the phase transition and then updated by EMA. The Student is the inference model.

## Evaluation protocol caveat

The current implementation computes Focal class weights from **the complete GT map**, including held-out pixels. It also computes OA/AA/Kappa on **all original-image pixels**, including labeled training pixels, and uses these full-image metrics for checkpoint selection. This is the exact behavior of the supplied training code, **not a strict held-out-pixel evaluation protocol**. `best_model.pth` maximizes OA; `best_model_composite.pth` maximizes `(OA+AA+Kappa)/3`. Do not claim test-only results or compare them as strictly equivalent to baselines using held-out masks without re-evaluation.
