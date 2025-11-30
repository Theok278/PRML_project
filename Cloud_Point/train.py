import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from model import build_sonata_classifier
from dataset import DigitsPointCloudDataset, collate_fn


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        # Move to device
        data_dict = {
            'coord': batch['coord'].to(device),
            'feat': batch['feat'].to(device),
            'offset': batch['offset'].to(device)
        }
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(data_dict)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch in pbar:
            # Move to device
            data_dict = {
                'coord': batch['coord'].to(device),
                'feat': batch['feat'].to(device),
                'offset': batch['offset'].to(device)
            }
            labels = batch['label'].to(device)

            # Forward pass
            logits = model(data_dict)
            loss = criterion(logits, labels)

            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def main(args):
    # Set device (support CUDA, MPS for Mac, and CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'sonata_digits_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')

    # Save arguments
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Load dataset
    print(f'Loading dataset from {args.data_dir}')
    full_dataset = DigitsPointCloudDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        normalization=True
    )

    # Split dataset
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f'Train size: {train_size}, Val size: {val_size}')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )

    # Build model
    print('Building SONATA classifier')
    model = build_sonata_classifier(
        num_classes=full_dataset.num_classes,
        pretrained_path=args.pretrained_path,
        freeze_encoder=args.freeze_encoder
    )
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if args.freeze_encoder:
        # Only optimize classification head
        optimizer = optim.Adam(
            model.head.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        # Optimize full model with different learning rates
        optimizer = optim.Adam([
            {'params': model.encoder.parameters(), 'lr': args.lr * 0.1},
            {'params': model.head.parameters(), 'lr': args.lr}
        ], weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # Training loop
    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print('\nStarting training...')
    for epoch in range(1, args.epochs + 1):
        print(f'\n{"=" * 50}')
        print(f'Epoch {epoch}/{args.epochs}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )

        # Update scheduler
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print summary
        print(f'\nEpoch {epoch} Summary:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'history': history
        }

        # Save last checkpoint
        torch.save(
            checkpoint,
            os.path.join(output_dir, 'last_checkpoint.pth')
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                checkpoint,
                os.path.join(output_dir, 'best_checkpoint.pth')
            )
            print(f'Best model saved with Val Acc: {val_acc:.2f}%')

        # Save history
        with open(os.path.join(output_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)

    print('\n' + '=' * 50)
    print('Training completed!')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SONATA for digit classification')

    # Data
    parser.add_argument('--data_dir', type=str,
                        default='../../digits_3d/training_data',
                        help='Path to training data directory')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length for point clouds (default: 128, based on data avg ~55 points)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')

    # Model
    parser.add_argument('--pretrained_path', type=str,
                        default='./sonata/pretrain-sonata-v1m1-0-base.pth',
                        help='Path to pretrained SONATA weights')
    parser.add_argument('--freeze_encoder', action='store_true', default=True,
                        help='Freeze encoder for transfer learning')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')

    # Others
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')

    args = parser.parse_args()
    main(args)
