# SONATA Transfer Learning for 3D Digit Classification

This project implements transfer learning using the SONATA (Self-Supervised Learning of Reliable Point Representations) model for 3D handwritten digit classification.

## Project Structure

```
Cloud_Point/
├── dataset.py          # Dataset loader for 3D digit point clouds
├── model.py            # SONATA encoder and classification head
├── train.py            # Training script
├── test.py             # Testing and evaluation script
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── sonata/             # Pretrained SONATA weights
│   ├── pretrain-sonata-v1m1-0-base.pth
│   └── ...
└── outputs/            # Training outputs (created automatically)
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Hardware Support

This project automatically detects and uses the best available hardware:

- **NVIDIA GPU (CUDA)**: Full support with CUDA acceleration
- **Apple Silicon (M1/M2/M3/M4)**: MPS (Metal Performance Shaders) GPU acceleration
- **CPU**: Fallback option (slower but always works)

The code automatically selects the best device. No manual configuration needed!

## Usage

### Training

Train the model with frozen encoder (transfer learning - only train classification head):

```bash
python train.py \
    --data_dir ../../digits_3d/training_data \
    --pretrained_path ./sonata/pretrain-sonata-v1m1-0-base.pth \
    --freeze_encoder \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --seq_len 128 \
    --val_split 0.2
```

### Testing

Evaluate the trained model:

```bash
python test.py \
    --checkpoint ./outputs/sonata_digits_XXXXXX/best_checkpoint.pth \
    --data_dir ../../digits_3d/training_data \
    --batch_size 32 \
    --seq_len 128
```

## Key Parameters

### Training Parameters

- `--data_dir`: Path to the training data directory containing CSV files
- `--pretrained_path`: Path to pretrained SONATA weights
- `--freeze_encoder`: Freeze encoder weights (recommended for transfer learning)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--seq_len`: Resample point clouds to this length (default: 512)
- `--val_split`: Validation split ratio (default: 0.2)
- `--weight_decay`: Weight decay for optimizer (default: 0.0001)

### Model Architecture

- **Encoder**: SONATA transformer-based encoder (frozen during transfer learning)
  - Embedding dimension: 384
  - Depth: 12 transformer layers
  - Number of heads: 6

- **Classification Head**: 2-layer MLP with dropout
  - Input: 384-dim global features
  - Hidden: 192 dims
  - Output: 10 classes (digits 0-9)

## Data Format

The dataset expects CSV files in the format:
```
x1,y1,z1
x2,y2,z2
...
```

Filename format: `stroke_LABEL_XXXX.csv` where LABEL is the digit (0-9).

## Preprocessing

The following preprocessing steps are applied:

1. **PCA Orientation**: Align point cloud using PCA
2. **Rectification**: Ensure first point is at the "top"
3. **Normalization**: Zero mean, unit variance
4. **Resampling**: Resample to fixed length (512 points)

## Outputs

Training outputs are saved to `./outputs/sonata_digits_TIMESTAMP/`:

- `best_checkpoint.pth`: Best model based on validation accuracy
- `last_checkpoint.pth`: Last epoch checkpoint
- `history.json`: Training history (loss and accuracy curves)
- `args.json`: Training arguments
- `test_results.json`: Test results (after running test.py)
- `confusion_matrix.png`: Confusion matrix visualization

## Transfer Learning Strategy

This implementation uses **feature extraction** approach:

1. Load pretrained SONATA encoder weights
2. Freeze encoder parameters
3. Train only the classification head
4. Use lower learning rate and fewer epochs

This approach is recommended when:
- Limited training data
- Task is related to the pretraining task
- Fast training is desired

## Advanced Usage

### Fine-tune the Full Model

To unfreeze and fine-tune the entire model:

```bash
python train.py \
    --data_dir ../../digits_3d/training_data \
    --pretrained_path ./sonata/pretrain-sonata-v1m1-0-base.pth \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0001 \
    --seq_len 128
```

Note: Remove `--freeze_encoder` flag to train the full model.

## References

- SONATA: Self-Supervised Learning of Reliable Point Representations
- Pointcept: https://github.com/Pointcept/Pointcept

## License

This code is for educational purposes. The SONATA pretrained weights follow CC-BY-NC 4.0 license.
