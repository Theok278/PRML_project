# 3D Air Writing Digit Classification with Transformer with numpy

A Transformer-based deep learning model for classifying 3D air-written digits (0-9) with manual backpropagation implementation.

## Project Structure

### Core Model Files
- `model.py` - Standard Transformer encoder implementation with manual backprop
- `model_supcon.py` - Transformer with Supervised Contrastive Learning support
- `modules.py` - Building blocks (Multi-head Attention, LayerNorm, MLP, SwiGLU, etc.)

### Training & Optimization
- `train_supcon.py` - Training script for SupCon variant
- `optimizer.py` - SGD and AdamW optimizers with manual implementation
- `scheduler.py` - Learning rate schedulers (WarmupCosine, CosineAnnealing)
- `loss.py` - Cross-entropy loss with manual backprop
- `supcon_loss.py` - Supervised Contrastive Loss

### Data Processing
- `dataset.py` - PyTorch Dataset for 3D stroke data with resampling & normalization
- `augmentation.py` - Data augmentation (rotation, scaling, jitter, time-warping) and movement feature extraction

### Hyperparameter Tuning
- `grid_search_supcon.py` - Grid search for SupCon model

### Inference & Evaluation
- `digit_classify.py` - Production inference API with automatic checkpoint loading
- `test_digit_classify.py` - Evaluation script with confusion matrix

## Quick Start

### Training
```bash
# Standard Transformer
bash train_mian.sh

# With Supervised Contrastive Learning
bash train_supcon.sh
```

### Inference
```python
from digit_classify import digit_classify

# Classify a single stroke
prediction = digit_classify("path/to/stroke.csv")   # numpy, list, turple, path supported
print(f"Predicted digit: {prediction}")
```

### Evaluation
```bash
python test_digit_classify.py
```

## Key Features

### Model Architecture
- **Transformer Encoder** with:
  - CLS token for global representation
  - Sinusoidal/Learnable/Conditional positional encoding
  - Multi-head self-attention
  - SwiGLU feed-forward networks
  - Pre-normalization and residual connections

### Data Processing
- **Resampling**: Fixed-length sequence (default 128 points)
- **Normalization**: Zero-mean, unit-variance per sample
- **Movement Features**: Optional velocity/acceleration features
- **Augmentation**: Rotation, scaling, translation, jitter, time-warping

### Training Techniques
- AdamW optimizer with weight decay
- Warmup + Cosine learning rate scheduling
- Label smoothing
- Supervised Contrastive Learning (optional)

## Performance

Current best model (detials in .sh file):
- **Architecture**: d_model=64, nhead=4, num_layers=2 
- **Training**: 20 epochs, AdamW optimizer, warmup_cosine scheduler

## Model Checkpoint

Trained models are saved in `outputs/*/best_checkpoint.npz` containing:
- Model parameters
- Training configuration (hyperparameters)
- Validation accuracy

Move the checkpoint from `outputs/*/best_checkpoint.npz` to `checkpoint/best_checkpoint.npz`
The inference API (`digit_classify.py`) automatically loads the best checkpoint.

## Data Format

Input CSV files should contain 3D points (x, y, z):
```
x1, y1, z1
x2, y2, z2
...
```

File naming convention: `stroke_<label>_<id>.csv`
- Example: `stroke_3_0042.csv` (digit 3, sample 42)

## Dependencies

- NumPy
- PyTorch (for data loading only)
- Pandas

## Implementation Notes

- **Manual Backpropagation**: All gradients computed manually (no autograd)
- **Modular Design**: Easy to swap components (attention, FFN, optimizer)
- **Production Ready**: Inference API handles all preprocessing automatically
