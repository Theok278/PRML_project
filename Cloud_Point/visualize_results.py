import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_training_history(history_path, output_dir):
    """Plot training history"""
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Training curves saved to {save_path}')

    # Print statistics
    print('\nTraining Statistics:')
    print(f'Best Train Accuracy: {max(history["train_acc"]):.2f}%')
    print(f'Best Val Accuracy: {max(history["val_acc"]):.2f}%')
    print(f'Final Train Loss: {history["train_loss"][-1]:.4f}')
    print(f'Final Val Loss: {history["val_loss"][-1]:.4f}')


def plot_per_class_metrics(results_path, output_dir):
    """Plot per-class metrics"""
    with open(results_path, 'r') as f:
        results = json.load(f)

    report = results['classification_report']

    digits = []
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(10):
        digit_str = str(i)
        if digit_str in report:
            digits.append(digit_str)
            precisions.append(report[digit_str]['precision'])
            recalls.append(report[digit_str]['recall'])
            f1_scores.append(report[digit_str]['f1-score'])

    x = np.arange(len(digits))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-score', alpha=0.8)

    ax.set_xlabel('Digit Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(digits)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'per_class_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Per-class metrics plot saved to {save_path}')


def main(args):
    if not os.path.exists(args.experiment_dir):
        print(f'Error: Experiment directory not found: {args.experiment_dir}')
        return

    print(f'Visualizing results from: {args.experiment_dir}')

    # Plot training history
    history_path = os.path.join(args.experiment_dir, 'history.json')
    if os.path.exists(history_path):
        print('\nPlotting training history...')
        plot_training_history(history_path, args.experiment_dir)
    else:
        print(f'Warning: history.json not found at {history_path}')

    # Plot test results
    results_path = os.path.join(args.experiment_dir, 'test_results.json')
    if os.path.exists(results_path):
        print('\nPlotting per-class metrics...')
        plot_per_class_metrics(results_path, args.experiment_dir)
    else:
        print(f'Info: test_results.json not found. Run test.py first to generate test results.')

    print('\nVisualization complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to experiment directory (e.g., ./outputs/sonata_digits_XXXXXX)')

    args = parser.parse_args()
    main(args)
