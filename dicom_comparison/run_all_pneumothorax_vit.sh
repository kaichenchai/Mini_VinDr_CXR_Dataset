#!/bin/bash

echo "Starting sequential training jobs..."

echo "=== Starting 2-bit training ==="
python3 training_script.py \
  --data_dir "/home/kai/mnt/VinDr_datasets/pneumothorax/2_bit_png_norm" \
  --model_name "vit_base_patch16_224" \
  --epochs 200 \
  --lr 0.001 \
  --num_chans 3 \
  --greyscale t \
  --batch_size 8 \
  --pretrained t \
  --run_name "pneumothorax_2_norm_200e_pretrained_vit" \
  --early_stop 50

echo "=== Finished 2-bit training ==="
echo "=== Starting 4-bit training ==="

python3 training_script.py \
  --data_dir "/home/kai/mnt/VinDr_datasets/pneumothorax/4_bit_png_norm" \
  --model_name "vit_base_patch16_224" \
  --epochs 200 \
  --lr 0.001 \
  --num_chans 3 \
  --greyscale t \
  --batch_size 8 \
  --pretrained t \
  --run_name "pneumothorax_4_norm_200e_pretrained_vit" \
  --early_stop 50

echo "=== Finished 4-bit training ==="
echo "=== Starting 8-bit training ==="

python3 training_script.py \
  --data_dir "/home/kai/mnt/VinDr_datasets/pneumothorax/8_bit_png_norm" \
  --model_name "vit_base_patch16_224" \
  --epochs 200 \
  --lr 0.001 \
  --num_chans 3 \
  --greyscale t \
  --batch_size 8 \
  --pretrained t \
  --run_name "pneumothorax_8_norm_200e_pretrained_vit" \
  --early_stop 50

echo "=== Finished 8-bit training ==="
echo "=== Starting 16-bit training ==="

python3 training_script.py \
  --data_dir "/home/kai/mnt/VinDr_datasets/pneumothorax/16_bit_png_norm" \
  --model_name "vit_base_patch16_224" \
  --epochs 200 \
  --lr 0.001 \
  --num_chans 3 \
  --greyscale t \
  --batch_size 8 \
  --pretrained t \
  --run_name "pneumothorax_16_norm_200e_pretrained_vit" \
  --early_stop 50

echo "=== Finished all training jobs ==="

