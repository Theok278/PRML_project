import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from model import build_sonata_classifier
from dataset import DigitsPointCloudDataset, collate_fn


def test_model(model, test_loader, device):
    """Test the model and compute metrics"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
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
            _, predicted = logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute accuracy
    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=[str(i) for i in range(10)],
        output_dict=True
    )

    return accuracy, cm, report, all_preds, all_labels


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(10),
        yticklabels=range(10)
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to {save_path}')


def main(args):
    # Set device (support CUDA, MPS for Mac, and CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Load dataset
    print(f'Loading test dataset from {args.data_dir}')
    test_dataset = DigitsPointCloudDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        normalization=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )

    print(f'Test dataset size: {len(test_dataset)}')

    # Load model
    print('Building model')
    model = build_sonata_classifier(
        num_classes=test_dataset.num_classes,
        pretrained_path=None,
        freeze_encoder=False
    )

    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f'Checkpoint epoch: {checkpoint.get("epoch", "unknown")}')
    print(f'Checkpoint val acc: {checkpoint.get("val_acc", "unknown")}')

    # Test
    print('\nTesting...')
    accuracy, cm, report, preds, labels = test_model(model, test_loader, device)

    # Print results
    print('\n' + '=' * 50)
    print(f'Test Accuracy: {accuracy:.2f}%')
    print('\nClassification Report:')
    print('=' * 50)
    for digit in range(10):
        digit_str = str(digit)
        if digit_str in report:
            metrics = report[digit_str]
            print(f'Digit {digit}: Precision={metrics["precision"]:.3f}, '
                  f'Recall={metrics["recall"]:.3f}, '
                  f'F1-score={metrics["f1-score"]:.3f}, '
                  f'Support={int(metrics["support"])}')

    print('\nOverall:')
    print(f'Macro avg: Precision={report["macro avg"]["precision"]:.3f}, '
          f'Recall={report["macro avg"]["recall"]:.3f}, '
          f'F1-score={report["macro avg"]["f1-score"]:.3f}')
    print(f'Weighted avg: Precision={report["weighted avg"]["precision"]:.3f}, '
          f'Recall={report["weighted avg"]["recall"]:.3f}, '
          f'F1-score={report["weighted avg"]["f1-score"]:.3f}')

    # Save results
    output_dir = os.path.dirname(args.checkpoint)
    results = {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f'\nResults saved to {results_path}')

    # Plot confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, cm_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test SONATA digit classifier')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str,
                        default='../../digits_3d/training_data',
                        help='Path to test data directory')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length for point clouds (default: 128, based on data avg ~55 points)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')

    args = parser.parse_args()
    main(args)
