#!/bin/bash

# Training script with AdamW + Warmup+Cosine scheduler

echo "=========================================="
echo "Training with AdamW + Warmup+Cosine"
echo "=========================================="

python train_manual_backprop.py \
    --data_dir ../../digits_3d/training_data \
    --optimizer adamw \
    --scheduler warmup_cosine \
    --epochs 20 \
    --batch_size 64 \
    --lr 0.001 \
    --weight_decay 0.01 \
    --warmup_epochs 2 \
    --min_lr 1e-6 \
    --d_model 128 \
    --nhead 8 \
    --num_layers 4 \
    --dim_feedforward 256 \
    --dropout 0.1 \
    --seq_len 128 \
    --val_split 0.2 \
    --seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "❌ Training failed!"
    exit 1
fi
