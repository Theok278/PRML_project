#!/bin/bash

# Quick start script for SONATA transfer learning on 3D digits

echo "================================================"
echo "SONATA Transfer Learning for 3D Digit Classification"
echo "================================================"

# Check if pretrained weights exist
PRETRAINED_PATH="./sonata/pretrain-sonata-v1m1-0-base.pth"
if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "Warning: Pretrained weights not found at $PRETRAINED_PATH"
    echo "Please ensure you have downloaded the SONATA pretrained weights"
    exit 1
fi

# Check if data directory exists
DATA_DIR="../../digits_3d/training_data"
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found at $DATA_DIR"
    echo "Please check the path to your training data"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Pretrained weights: $PRETRAINED_PATH"
echo "  Batch size: 32"
echo "  Epochs: 10"
echo "  Learning rate: 0.001"
echo "  Sequence length: 128"
echo "  Validation split: 0.2"
echo "  Transfer learning: Yes (encoder frozen)"
echo ""

read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Training cancelled"
    exit 0
fi

# Start training
python train.py \
    --data_dir "$DATA_DIR" \
    --pretrained_path "$PRETRAINED_PATH" \
    --freeze_encoder \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.001 \
    --seq_len 128 \
    --val_split 0.2 \
    --num_workers 4 \
    --output_dir ./outputs

echo ""
echo "Training completed!"
echo "Check ./outputs/ for results"
