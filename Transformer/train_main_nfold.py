import numpy as np
import argparse
import json
import time
from pathlib import Path
from typing import Tuple
from datetime import datetime

from model import Transformer
from optimizer import SGD, AdamW
from scheduler import WarmupCosineScheduler, CosineAnnealingScheduler
from loss import CrossEntropyLoss

import torch
from torch.utils.data import DataLoader
from dataset import DigitsStrokeDataset


def numpy_from_dataloader(dataloader) -> Tuple[np.ndarray, np.ndarray]:
    """convert PyTorch dataloader batch to numpy arrays"""

    all_data = []
    all_labels = []

    for batch_data, batch_labels in dataloader:
        # convert to numpy
        all_data.append(batch_data.numpy())
        all_labels.append(batch_labels.numpy())

    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return data, labels


def evaluate(model: Transformer, data: np.ndarray, labels: np.ndarray,
             batch_size: int = 32) -> Tuple[float, float]:
    """evaluate model on dataset"""
    model.eval()

    num_samples = data.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    total_loss = 0.0
    correct = 0

    criterion = CrossEntropyLoss()

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_data = data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        # forward pass (no gradient)
        logits = model.forward(batch_data, training=False)

        # loss
        loss = criterion.forward(logits, batch_labels)
        total_loss += loss * (end_idx - start_idx)

        # accuracy
        preds = np.argmax(logits, axis=1)
        correct += (preds == batch_labels).sum()

    avg_loss = total_loss / num_samples
    accuracy = correct / num_samples

    return avg_loss, accuracy


def load_pretrained_weights(model: Transformer, checkpoint_path: str, strict: bool = False):
    """
    Load pretrained weights from checkpoint

    Args:
        model: The model to load weights into
        checkpoint_path: Path to .npz checkpoint file
        strict: If True, require exact parameter match. If False, allow partial loading

    Returns:
        Dictionary with loading statistics
    """
    print(f"\nLoading pretrained weights from: {checkpoint_path}")

    try:
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}")

    # Get model parameters
    model_params = list(model.parameters())

    # Count parameters in checkpoint
    checkpoint_param_keys = [k for k in checkpoint.keys() if k.startswith('param_')]
    num_checkpoint_params = len(checkpoint_param_keys)
    num_model_params = len(model_params)

    print(f"  Checkpoint has {num_checkpoint_params} parameters")
    print(f"  Current model has {num_model_params} parameters")

    if strict and num_checkpoint_params != num_model_params:
        raise ValueError(
            f"Parameter count mismatch: checkpoint has {num_checkpoint_params}, "
            f"model has {num_model_params}. Use --strict_load=False to allow partial loading."
        )

    # Load parameters
    loaded_count = 0
    skipped_count = 0
    shape_mismatch_count = 0

    for i, param in enumerate(model_params):
        param_key = f'param_{i}'

        if param_key not in checkpoint:
            if strict:
                raise ValueError(f"Missing parameter: {param_key}")
            skipped_count += 1
            continue

        checkpoint_param = checkpoint[param_key]

        # Check shape compatibility
        if checkpoint_param.shape != param.data.shape:
            if strict:
                raise ValueError(
                    f"Shape mismatch for {param_key}: "
                    f"checkpoint has {checkpoint_param.shape}, model has {param.data.shape}"
                )
            shape_mismatch_count += 1
            continue

        # Load the parameter
        param.data[:] = checkpoint_param
        loaded_count += 1

    # Print loading summary
    print(f"\n  ✓ Loaded {loaded_count}/{num_model_params} parameters")
    if skipped_count > 0:
        print(f"  ⚠ Skipped {skipped_count} parameters (not in checkpoint)")
    if shape_mismatch_count > 0:
        print(f"  ⚠ Skipped {shape_mismatch_count} parameters (shape mismatch)")

    # Print checkpoint metadata if available
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch'].item() if hasattr(checkpoint['epoch'], 'item') else checkpoint['epoch']
        print(f"\n  Checkpoint info:")
        print(f"    - Epoch: {epoch}")
    if 'val_acc' in checkpoint:
        val_acc = checkpoint['val_acc'].item() if hasattr(checkpoint['val_acc'], 'item') else checkpoint['val_acc']
        print(f"    - Val Accuracy: {val_acc:.4f}")

    return {
        'loaded': loaded_count,
        'skipped': skipped_count,
        'shape_mismatch': shape_mismatch_count,
        'total': num_model_params
    }


def freeze_parameters(model: Transformer, freeze_encoder: bool = False, freeze_layers: int = None):
    """
    Freeze model parameters to prevent updates during training

    Args:
        model: The model
        freeze_encoder: If True, freeze all encoder layers (keep only head trainable)
        freeze_layers: Number of bottom encoder layers to freeze (0-indexed)
    """
    if freeze_encoder:
        print("\n  Freezing all encoder layers (only training classification head)")
        # Freeze everything except the classification head
        for param in model.input_proj.parameters():
            param.requires_grad = False
        model.cls_token.requires_grad = False
        for param in model.pos_encoder.parameters():
            param.requires_grad = False
        for layer in model.layers:
            for param in layer.parameters():
                param.requires_grad = False
        for param in model.norm.parameters():
            param.requires_grad = False
        # Keep head trainable
        trainable_params = sum(p.data.size for p in model.head.parameters())
        total_params = sum(p.data.size for p in model.parameters())
        print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100*trainable_params/total_params:.1f}%)")

    elif freeze_layers is not None:
        if freeze_layers <= 0:
            print(f"\n  No layers frozen (freeze_layers={freeze_layers})")
            return

        num_layers = len(model.layers)
        freeze_layers = min(freeze_layers, num_layers)
        print(f"\n  Freezing bottom {freeze_layers}/{num_layers} encoder layers")

        # Freeze specified number of bottom layers
        for i in range(freeze_layers):
            for param in model.layers[i].parameters():
                param.requires_grad = False

        # Count trainable parameters
        trainable_params = sum(p.data.size for p in model.parameters() if p.requires_grad)
        total_params = sum(p.data.size for p in model.parameters())
        print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100*trainable_params/total_params:.1f}%)")


def train_epoch(model: Transformer, optimizer: SGD,
                train_data: np.ndarray, train_labels: np.ndarray,
                batch_size: int = 32) -> Tuple[float, float]:
    """train for one epoch"""
    model.train()

    num_samples = train_data.shape[0]
    indices = np.random.permutation(num_samples)

    num_batches = (num_samples + batch_size - 1) // batch_size

    total_loss = 0.0
    correct = 0

    criterion = CrossEntropyLoss()

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_indices = indices[start_idx:end_idx]
        batch_data = train_data[batch_indices]
        batch_labels = train_labels[batch_indices]

        # zero gradients
        optimizer.zero_grad()

        # forward pass
        logits = model.forward(batch_data, training=True)

        # compute loss
        loss = criterion.forward(logits, batch_labels)
        total_loss += loss * (end_idx - start_idx)

        # accuracy
        preds = np.argmax(logits, axis=1)
        correct += (preds == batch_labels).sum()

        # backward pass (manual!)
        grad_logits = criterion.backward()
        model.backward(grad_logits)

        # update parameters
        optimizer.step()

    avg_loss = total_loss / num_samples
    accuracy = correct / num_samples

    return avg_loss, accuracy


def create_model(args, input_dim):
    """Create and initialize a model"""
    model = Transformer(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        num_classes=10,
        pos_encoding=args.pos_encoding
    )

    # Load pretrained weights if specified
    if args.pretrained:
        load_pretrained_weights(model, args.pretrained, strict=args.strict_load)

        # Apply freezing if requested
        if args.freeze_encoder or args.freeze_layers is not None:
            freeze_parameters(model, freeze_encoder=args.freeze_encoder,
                            freeze_layers=args.freeze_layers)

    return model


def create_optimizer(args, model):
    """Create optimizer based on args"""
    if args.optimizer == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:  # adamw
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    return optimizer


def create_scheduler(args, optimizer):
    """Create learning rate scheduler based on args"""
    if args.scheduler == 'warmup_cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.epochs,
            base_lr=args.lr,
            min_lr=args.min_lr
        )
    elif args.scheduler == 'cosine':
        eta_min = args.eta_min if args.eta_min is not None else args.lr * 0.01
        scheduler = CosineAnnealingScheduler(
            optimizer,
            T_max=args.epochs,
            base_lr=args.lr,
            eta_min=eta_min
        )
    else:
        scheduler = None
    return scheduler


def train_fold(fold, train_files, val_files, args, output_dir, input_dim):
    """Train a single fold"""
    print("\n" + "="*60)
    print(f"FOLD {fold + 1}/{args.n_folds}")
    print("="*60)

    # Handle augmentation and movement features arguments
    augmentation = args.augmentation if args.augmentation != 'none' else None
    movement_features = args.movement_features if args.movement_features != 'none' else None

    # Create datasets for this fold
    train_dataset = DigitsStrokeDataset(
        args.data_dir,
        seq_len=args.seq_len,
        file_list=train_files,
        augmentation=augmentation,
        movement_features=movement_features,
        training=True
    )
    val_dataset = DigitsStrokeDataset(
        args.data_dir,
        seq_len=args.seq_len,
        file_list=val_files,
        augmentation=None,
        movement_features=movement_features,
        training=False
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    # Convert to numpy
    print("Converting data to numpy...")
    train_data, train_labels = numpy_from_dataloader(train_loader)
    val_data, val_labels = numpy_from_dataloader(val_loader)
    print(f"Train: {train_data.shape}, Val: {val_data.shape}")

    # Create model
    model = create_model(args, input_dim)
    num_params = sum(p.data.size for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(args, model)
    scheduler = create_scheduler(args, optimizer)

    # Training loop
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Update learning rate
        if scheduler is not None:
            current_lr = scheduler.step(epoch)
        else:
            current_lr = args.lr

        # Train
        train_loss, train_acc = train_epoch(
            model, optimizer, train_data, train_labels, args.batch_size
        )

        # Validate
        val_loss, val_acc = evaluate(
            model, val_data, val_labels, args.batch_size
        )

        epoch_time = time.time() - epoch_start

        # Save history
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['lr'].append(float(current_lr))

        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) - "
              f"LR: {current_lr:.6f}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model for this fold
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'fold': fold,
                'val_acc': val_acc,
                'args': json.dumps(vars(args))
            }
            for i, p in enumerate(model.parameters()):
                checkpoint[f'param_{i}'] = p.data.copy()

            fold_dir = output_dir / f'fold_{fold+1}'
            fold_dir.mkdir(parents=True, exist_ok=True)
            np.savez(fold_dir / 'best_checkpoint.npz', **checkpoint)
            print(f"  → Saved best model (val_acc: {val_acc:.4f})")

    # Save final model for this fold
    checkpoint = {
        'epoch': args.epochs - 1,
        'fold': fold,
        'val_acc': val_acc,
        'args': json.dumps(vars(args))
    }
    for i, p in enumerate(model.parameters()):
        checkpoint[f'param_{i}'] = p.data.copy()

    fold_dir = output_dir / f'fold_{fold+1}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.savez(fold_dir / 'last_checkpoint.npz', **checkpoint)

    # Save history for this fold
    with open(fold_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nFold {fold + 1} complete! Best val accuracy: {best_val_acc:.4f}")

    return best_val_acc, history


def main():
    parser = argparse.ArgumentParser(description='Train Transformer with N-Fold Cross Validation')
    parser.add_argument('--data_dir', type=str, default='../../digits_3d/training_data',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--mlp_ratio', type=float, default=None, help='MLP ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # N-fold cross validation
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross validation')

    # optimizer and scheduler options
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='warmup_cosine',
                        choices=['warmup_cosine', 'cosine', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs (for warmup_cosine)')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--eta_min', type=float, default=None, help='Minimum lr for cosine (default: lr * 0.01)')

    # augmentation options
    parser.add_argument('--augmentation', type=str, default=None,
                        choices=['light', 'medium', 'strong', 'none'],
                        help='Data augmentation strength (default: None)')
    parser.add_argument('--movement_features', type=str, default=None,
                        choices=['concat', 'replace', 'all', 'velocity_acceleration', 'acceleration_only', 'none'],
                        help='Movement feature extraction (default: None)')

    # model architecture options
    parser.add_argument('--pos_encoding', type=str, default='sinusoidal',
                        choices=['sinusoidal', 'learnable', 'conditional'],
                        help='Positional encoding type (default: sinusoidal)')

    # pretrained weights options
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained checkpoint (.npz file)')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder layers (only train classification head)')
    parser.add_argument('--freeze_layers', type=int, default=None,
                        help='Number of encoder layers to freeze (from bottom)')
    parser.add_argument('--strict_load', action='store_true',
                        help='Require exact parameter match when loading (default: allow partial loading)')

    args = parser.parse_args()

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f'outputs/nfold_ce_{args.n_folds}fold_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # load data
    print("\nLoading data...")

    # Handle augmentation and movement features arguments
    movement_features = args.movement_features if args.movement_features != 'none' else None

    # Create full dataset to get file list
    full_dataset = DigitsStrokeDataset(
        args.data_dir,
        seq_len=args.seq_len,
        augmentation=None,
        movement_features=None,
        training=False
    )

    # Get input dimension
    from augmentation import get_input_dim
    input_dim = get_input_dim(movement_features)

    print(f"Total samples: {len(full_dataset)}")
    print(f"Number of folds: {args.n_folds}")
    print(f"Input dimension: {input_dim}")
    print(f"Movement features: {movement_features if movement_features else 'None'}")
    print(f"Positional encoding: {args.pos_encoding}")

    # Create N-fold splits
    indices = np.random.permutation(len(full_dataset))
    fold_indices = np.array_split(indices, args.n_folds)

    # Store results for all folds
    all_fold_results = []

    # Train each fold
    for fold in range(args.n_folds):
        # Create train/val split for this fold
        val_idx = fold_indices[fold]
        train_idx = np.concatenate([fold_indices[i] for i in range(args.n_folds) if i != fold])

        train_files = [full_dataset.files[i] for i in train_idx]
        val_files = [full_dataset.files[i] for i in val_idx]

        print(f"\nFold {fold + 1}: Train samples: {len(train_files)}, Val samples: {len(val_files)}")

        # Train this fold
        best_val_acc, history = train_fold(
            fold, train_files, val_files, args, output_dir, input_dim
        )

        all_fold_results.append({
            'fold': fold + 1,
            'best_val_acc': best_val_acc,
            'final_val_acc': history['val_acc'][-1],
            'history': history
        })

    # Compute cross-validation statistics
    fold_accs = [r['best_val_acc'] for r in all_fold_results]
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    # Save cross-validation results
    cv_results = {
        'n_folds': args.n_folds,
        'fold_accuracies': fold_accs,
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'all_folds': all_fold_results
    }

    with open(output_dir / 'cv_results.json', 'w') as f:
        json.dump(cv_results, f, indent=2)

    # Print final results
    print("\n" + "="*60)
    print("N-FOLD CROSS VALIDATION COMPLETE!")
    print("="*60)
    print(f"Number of folds: {args.n_folds}")
    print(f"\nFold accuracies:")
    for i, acc in enumerate(fold_accs):
        print(f"  Fold {i+1}: {acc:.4f}")
    print(f"\nMean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print("="*60)
    print("\n✅ All gradients were computed MANUALLY")
    print("✅ No PyTorch autograd was used!")
    print("="*60)


if __name__ == '__main__':
    main()
