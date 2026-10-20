#!/bin/bash
# S13-A1: S3-A psi-guided soft perturbation target + S1-A entropy-guided
# noise strength scale, Farm 20%.
#
# Usage: bash run_v4_s13a1_farm20.sh <gpu_id> <seed>

GPU=${1:-0}
SEED=${2:-2333}
PYTHON=/root/anaconda3/envs/MambaHSI_env/bin/python
FARM_MAT=/root/Documents/WPIP_Mamba_CD/dataset/farm_China_Multitype/China_Change_Dataset.mat
FARM_GT=/root/Documents/WPIP_Mamba_CD/dataset/farm_China_Multitype/GT_farm_2D.mat

cd /root/Documents/SemiHSCD

${PYTHON} train.py \
    --dataset farm \
    --t1t2_mat ${FARM_MAT} --gt_mat ${FARM_GT} \
    --split_ratio 0.20 \
    --log_dir ./logs/v4_s13a1_entropy_min05_max20_s3a_soft_floor05_farm_20pct_seed${SEED} \
    --model v4 --gpu ${GPU} --seed ${SEED} \
    --d_state 16 --mamba_expand 1 --spec_d_inner 32 \
    --num_blocks 1 2 2 2 \
    --total_steps 2000 --sl_steps 400 \
    --log_interval 200 --eval_interval 200 \
    --eta 0.25 --eta_proto 0.3 --proto_momentum 0.9 \
    --ssl_aug_sigma 0.1 --class0_weight_floor 0.5 \
    --perturb_target_mode psi_guided_soft \
    --adaptive_noise_enabled \
    --adaptive_noise_uncertainty_source entropy \
    --adaptive_noise_min_scale 0.5 --adaptive_noise_max_scale 2.0
echo "[Done] Farm 20% S13-A1 psi_guided_soft + entropy_scale min=0.5 max=2.0 seed=${SEED}"
