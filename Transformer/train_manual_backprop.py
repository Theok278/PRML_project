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


def main():
    parser = argparse.ArgumentParser(description='Train Transformer with manual backprop')
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
    parser.add_argument('--dim_feedforward', type=int, default=256, help='FFN dimension')
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
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs (for warmup_cosine)')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--eta_min', type=float, default=None, help='Minimum lr for cosine (default: lr * 0.01)')

    args = parser.parse_args()

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f'outputs/manual_backprop_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # load data using PyTorch DataLoader (only for convenience)
    print("\nLoading data...")
    full_dataset = DigitsStrokeDataset(args.data_dir, seq_len=args.seq_len)

    # split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
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

    # create model (pure NumPy, no PyTorch)
    print("\nCreating model...")
    model = Transformer(
        input_dim=3,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_classes=10
    )

    # count parameters
    num_params = sum(p.data.size for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

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
        train_loss, train_acc = train_epoch(
            model, optimizer, train_data, train_labels, args.batch_size
        )

        # validate
        val_loss, val_acc = evaluate(
            model, val_data, val_labels, args.batch_size
        )

        epoch_time = time.time() - epoch_start

        # save history
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['lr'].append(float(current_lr))

        # print progress
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) - "
              f"LR: {current_lr:.6f}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # save model parameters (each parameter separately)
            checkpoint = {
                'epoch': epoch,
                'val_acc': val_acc,
                'args': json.dumps(vars(args))
            }
            # add each parameter with a unique key
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
    # add each parameter with a unique key
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
    print("✅ No PyTorch autograd was used!")
    print("="*60)


if __name__ == '__main__':
    main()
