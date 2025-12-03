import numpy as np
import argparse
import json
import time
from pathlib import Path
from typing import Tuple
from datetime import datetime

from model import Transformer
from model_supcon import TransformerSupCon
from optimizer import SGD, AdamW
from scheduler import WarmupCosineScheduler, CosineAnnealingScheduler
from loss import CrossEntropyLoss
from supcon_loss import SupConLoss

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


def evaluate(model,
             data: np.ndarray,
             labels: np.ndarray,
             batch_size: int = 32,
             is_supcon: bool = False) -> Tuple[float, float]:
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
        if is_supcon:
            # SupCon 模型：只取 logits（不需要 embeddings）
            logits = model.forward(batch_data, training=False, return_embeddings=False)
        else:
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


def load_pretrained_weights(model, checkpoint_path: str, strict: bool = False):
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


def freeze_parameters(model,
                      freeze_encoder: bool = False,
                      freeze_layers: int = None):
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


def train_epoch(model,
                optimizer,
                train_data: np.ndarray,
                train_labels: np.ndarray,
                batch_size: int = 32) -> Tuple[float, float]:
    """train for one epoch (CE-only)"""
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


def train_epoch_supcon(model: TransformerSupCon,
                       optimizer,
                       train_data: np.ndarray,
                       train_labels: np.ndarray,
                       supcon_loss: SupConLoss,
                       ce_loss: CrossEntropyLoss,
                       supcon_weight: float,
                       batch_size: int = 32) -> dict:
    """
    Train for one epoch with SupCon + Classification (joint training)

    supcon_weight in [0,1]:
        0.0 → 仅 CE（其实就退化成普通 CE）
        0.5 → SupCon 和 CE 各一半
        1.0 → 仅 SupCon（没有 CE）
    """
    model.train()

    num_samples = train_data.shape[0]
    indices = np.random.permutation(num_samples)
    num_batches = (num_samples + batch_size - 1) // batch_size

    total_loss = 0.0
    total_supcon_loss = 0.0
    total_ce_loss = 0.0
    correct = 0

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_indices = indices[start_idx:end_idx]
        batch_data = train_data[batch_indices]
        batch_labels = train_labels[batch_indices]

        # zero gradients
        optimizer.zero_grad()

        # forward pass: 得到 embeddings + logits
        embeddings, logits = model.forward(
            batch_data,
            training=True,
            return_embeddings=True
        )

        # 计算损失
        supcon_val = supcon_loss.forward(embeddings, batch_labels)
        ce_val = ce_loss.forward(logits, batch_labels)

        loss = supcon_weight * supcon_val + (1.0 - supcon_weight) * ce_val

        total_loss += loss * (end_idx - start_idx)
        total_supcon_loss += supcon_val * (end_idx - start_idx)
        total_ce_loss += ce_val * (end_idx - start_idx)

        # 分类 accuracy
        preds = np.argmax(logits, axis=1)
        correct += (preds == batch_labels).sum()

        # backward（手写梯度）
        grad_embeddings = supcon_loss.backward() * supcon_weight
        grad_logits = ce_loss.backward() * (1.0 - supcon_weight)

        model.backward(
            grad_embeddings=grad_embeddings,
            grad_logits=grad_logits
        )

        # update
        optimizer.step()

    avg_loss = total_loss / num_samples
    avg_supcon_loss = total_supcon_loss / num_samples
    avg_ce_loss = total_ce_loss / num_samples
    accuracy = correct / num_samples

    return {
        'loss': float(avg_loss),
        'supcon_loss': float(avg_supcon_loss),
        'ce_loss': float(avg_ce_loss),
        'accuracy': float(accuracy),
    }


def main():
    parser = argparse.ArgumentParser(description='Train Transformer (manual backprop, CE / SupCon)')

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
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # optimizer and scheduler options
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='warmup_cosine',
                        choices=['warmup_cosine', 'cosine', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs (for warmup_cosine)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--eta_min', type=float, default=None,
                        help='Minimum lr for cosine (default: lr * 0.01)')

    # augmentation options
    parser.add_argument('--augmentation', type=str, default=None,
                        choices=['light', 'medium', 'strong', 'none'],
                        help='Data augmentation strength (default: None)')
    parser.add_argument('--movement_features', type=str, default=None,
                        choices=['concat', 'replace', 'all',
                                 'velocity_acceleration', 'acceleration_only', 'none'],
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

    # SupCon options
    parser.add_argument('--supcon_weight', type=float, default=0.0,
                        help='Weight for SupCon loss (0=CE only, 0.5=joint, 1=SupCon only)')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for SupCon loss')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Projection head output dimension (SupCon)')
    parser.add_argument('--projection_hidden_dim', type=int, default=256,
                        help='Projection head hidden dimension (SupCon)')

    args = parser.parse_args()

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 是否启用 SupCon
    use_supcon = args.supcon_weight > 0.0

    # create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "supcon" if use_supcon else "ce"
    output_dir = Path(f'outputs/manual_backprop_{tag}_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # load data using PyTorch DataLoader (only for convenience)
    print("\nLoading data...")

    # Handle augmentation and movement features arguments
    augmentation = args.augmentation if args.augmentation != 'none' else None
    movement_features = args.movement_features if args.movement_features != 'none' else None

    # Create full dataset to get file list
    full_dataset = DigitsStrokeDataset(
        args.data_dir,
        seq_len=args.seq_len,
        augmentation=None,
        movement_features=None,
        training=False
    )

    # split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    # Get train/val file lists
    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_files = [full_dataset.files[i] for i in train_indices]
    val_files = [full_dataset.files[i] for i in val_indices]

    # Create separate datasets with/without augmentation
    train_dataset = DigitsStrokeDataset(
        args.data_dir,
        seq_len=args.seq_len,
        file_list=train_files,
        augmentation=augmentation,            # Use augmentation for training
        movement_features=movement_features,  # Use movement features for training
        training=True
    )
    val_dataset = DigitsStrokeDataset(
        args.data_dir,
        seq_len=args.seq_len,
        file_list=val_files,
        augmentation=None,                   # No augmentation for validation
        movement_features=movement_features, # But keep movement features for validation
        training=False
    )

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # convert to numpy arrays (do this once to avoid repeated conversion)
    print("Converting data to numpy...")
    train_data, train_labels = numpy_from_dataloader(train_loader)
    val_data, val_labels = numpy_from_dataloader(val_loader)

    print(f"Train: {train_data.shape}, Val: {val_data.shape}")

    # Determine input dimension based on movement features
    from augmentation import get_input_dim
    input_dim = get_input_dim(movement_features)

    # create model
    print("\nCreating model...")
    print(f"  Input dimension: {input_dim}")
    print(f"  Movement features: {movement_features if movement_features else 'None'}")
    print(f"  Positional encoding: {args.pos_encoding}")
    if use_supcon:
        print(f"  Training objective: SupCon + CE (weight={args.supcon_weight})")
        print(f"  Projection dim: {args.projection_dim}, hidden dim: {args.projection_hidden_dim}")
    else:
        print(f"  Training objective: CE only")

    if use_supcon:
        model = TransformerSupCon(
            input_dim=input_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            num_classes=10,
            pos_encoding=args.pos_encoding,
            projection_dim=args.projection_dim,
            projection_hidden_dim=args.projection_hidden_dim
        )
    else:
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

    # count parameters
    num_params = sum(p.data.size for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Load pretrained weights if specified
    if args.pretrained:
        load_pretrained_weights(model, args.pretrained, strict=args.strict_load)

        # Apply freezing if requested
        if args.freeze_encoder or args.freeze_layers is not None:
            freeze_parameters(model, freeze_encoder=args.freeze_encoder,
                              freeze_layers=args.freeze_layers)

    # create optimizer
    if args.optimizer == 'sgd':
        print(f"\nUsing SGD optimizer (lr={args.lr}, momentum={args.momentum})")
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:  # adamw
        print(f"\nUsing AdamW optimizer (lr={args.lr}, weight_decay={args.weight_decay})")
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    # create learning rate scheduler
    if args.scheduler == 'warmup_cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.epochs,
            base_lr=args.lr,
            min_lr=args.min_lr
        )
        print(f"Using Warmup + Cosine scheduler (warmup={args.warmup_epochs} epochs, min_lr={args.min_lr})")
    elif args.scheduler == 'cosine':
        eta_min = args.eta_min if args.eta_min is not None else args.lr * 0.01
        scheduler = CosineAnnealingScheduler(
            optimizer,
            T_max=args.epochs,
            base_lr=args.lr,
            eta_min=eta_min
        )
        print(f"Using Cosine Annealing scheduler (T_max={args.epochs}, eta_min={eta_min})")
    else:
        scheduler = None
        print("No learning rate scheduler")

    # SupCon 损失（如果启用）
    if use_supcon:
        supcon_loss = SupConLoss(temperature=args.temperature)
        ce_loss = CrossEntropyLoss()

    # training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

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

        # update learning rate with scheduler
        if scheduler is not None:
            current_lr = scheduler.step(epoch)
        else:
            current_lr = args.lr

        # train
        if use_supcon:
            train_stats = train_epoch_supcon(
                model=model,
                optimizer=optimizer,
                train_data=train_data,
                train_labels=train_labels,
                supcon_loss=supcon_loss,
                ce_loss=ce_loss,
                supcon_weight=args.supcon_weight,
                batch_size=args.batch_size
            )
            train_loss = train_stats['loss']
            train_acc = train_stats['accuracy']
        else:
            train_loss, train_acc = train_epoch(
                model, optimizer, train_data, train_labels, args.batch_size
            )

        # validate
        val_loss, val_acc = evaluate(
            model, val_data, val_labels, args.batch_size, is_supcon=use_supcon
        )

        epoch_time = time.time() - epoch_start

        # save history
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['lr'].append(float(current_lr))

        # print progress
        if use_supcon:
            print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) - "
                  f"LR: {current_lr:.6f}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) - "
                  f"LR: {current_lr:.6f}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'val_acc': val_acc,
                'args': json.dumps(vars(args))
            }
            for i, p in enumerate(model.parameters()):
                checkpoint[f'param_{i}'] = p.data.copy()

            np.savez(output_dir / 'best_checkpoint.npz', **checkpoint)
            print(f"  → Saved best model (val_acc: {val_acc:.4f})")

    # save final model
    checkpoint = {
        'epoch': args.epochs - 1,
        'val_acc': val_acc,
        'args': json.dumps(vars(args))
    }
    for i, p in enumerate(model.parameters()):
        checkpoint[f'param_{i}'] = p.data.copy()

    np.savez(output_dir / 'last_checkpoint.npz', **checkpoint)

    # save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    print("\n✅ All gradients were computed MANUALLY")
    if use_supcon:
        print("✅ Trained with Supervised Contrastive Learning (SupCon + CE)")
    else:
        print("✅ Trained with pure Cross-Entropy")
    print("✅ No PyTorch autograd was used!")
    print("="*60)


if __name__ == '__main__':
    main()