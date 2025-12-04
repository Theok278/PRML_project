#!/bin/bash

echo "=========================================="
echo "Training start"
echo "=========================================="

python train_supcon.py \
    --data_dir ../../digits_3d/training_data \
    --augmentation light \
    --movement_features none \
    --optimizer adamw \
    --scheduler warmup_cosine \
    --pos_encoding sinusoidal \
    --epochs 20 \
    --batch_size 16 \
    --lr 0.001 \
    --weight_decay 0.05 \
    --warmup_epochs 2 \
    --min_lr 1e-6 \
    --d_model 64 \
    --nhead 4 \
    --num_layers 2 \
    --dropout 0.1 \
    --seq_len 128 \
    --val_split 0.2 \
    --supcon_weight 0.5 \
    --temperature 0.05 \
    --projection_dim 32 \
    --projection_hidden_dim 32 \
    --seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "Training failed!"
    exit 1
fi
