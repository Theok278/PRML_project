"""
Grid Search with K-Fold Cross Validation for Hyperparameter Tuning
(对齐 manual_backprop 训练脚本版本，支持 SupCon / TransformerSupCon)
"""

import numpy as np
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from itertools import product
import glob
import os

from model import Transformer
from model_supcon import TransformerSupCon
from optimizer import SGD, AdamW
from scheduler import WarmupCosineScheduler, CosineAnnealingScheduler
from loss import CrossEntropyLoss
from supcon_loss import SupConLoss

import torch
from torch.utils.data import DataLoader
from dataset import DigitsStrokeDataset
from augmentation import get_input_dim


# ------------------------------
# K-fold 切分（和数据集文件命名一致）
# ------------------------------
def create_k_folds(file_list: List[str], labels: List[int],
                   n_folds: int = 5, seed: int = 42) -> List[Tuple[List[str], List[str]]]:
    """
    Create K-fold splits stratified by class

    Args:
        file_list: List of file paths
        labels: List of labels corresponding to files
        n_folds: Number of folds
        seed: Random seed

    Returns:
        List of (train_files, val_files) tuples for each fold
    """
    np.random.seed(seed)

    # Group files by label
    label_to_files: Dict[int, List[str]] = {}
    for file, label in zip(file_list, labels):
        if label not in label_to_files:
            label_to_files[label] = []
        label_to_files[label].append(file)

    # Shuffle each class
    for label in label_to_files:
        np.random.shuffle(label_to_files[label])

    # Create folds
    folds: List[List[str]] = [[] for _ in range(n_folds)]

    # Distribute each class across folds
    for label, files in label_to_files.items():
        n_samples = len(files)
        fold_sizes = [n_samples // n_folds] * n_folds
        for i in range(n_samples % n_folds):
            fold_sizes[i] += 1

        start_idx = 0
        for fold_idx, fold_size in enumerate(fold_sizes):
            folds[fold_idx].extend(files[start_idx:start_idx + fold_size])
            start_idx += fold_size

    splits: List[Tuple[List[str], List[str]]] = []
    for val_fold_idx in range(n_folds):
        val_files = folds[val_fold_idx]
        train_files: List[str] = []
        for train_fold_idx in range(n_folds):
            if train_fold_idx != val_fold_idx:
                train_files.extend(folds[train_fold_idx])
        splits.append((train_files, val_files))

    return splits


# ------------------------------
# 和训练脚本一致的工具函数
# ------------------------------
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
            # SupCon 模型：只要 logits，不要 embeddings
            logits = model.forward(batch_data, training=False, return_embeddings=False)
        else:
            # 普通 Transformer
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


def train_epoch(model: Transformer, optimizer,
                train_data: np.ndarray, train_labels: np.ndarray,
                batch_size: int = 32) -> Tuple[float, float]:
    """train for one epoch (和训练脚本一致，不在这里用 scheduler)"""
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


def train_epoch_supcon(
    model: TransformerSupCon,
    optimizer,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    supcon_loss: SupConLoss,
    ce_loss: CrossEntropyLoss,
    supcon_weight: float,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Train for one epoch with SupCon + Classification (manual backprop)
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

        # forward: embeddings + logits
        embeddings, logits = model.forward(
            batch_data,
            training=True,
            return_embeddings=True
        )

        # losses
        supcon_val = supcon_loss.forward(embeddings, batch_labels)
        ce_val = ce_loss.forward(logits, batch_labels)

        loss = supcon_weight * supcon_val + (1.0 - supcon_weight) * ce_val

        total_loss += loss * (end_idx - start_idx)
        total_supcon_loss += supcon_val * (end_idx - start_idx)
        total_ce_loss += ce_val * (end_idx - start_idx)

        # acc
        preds = np.argmax(logits, axis=1)
        correct += (preds == batch_labels).sum()

        # backward (manual)
        grad_embeddings = supcon_loss.backward() * supcon_weight
        grad_logits = ce_loss.backward() * (1.0 - supcon_weight)

        model.backward(
            grad_embeddings=grad_embeddings,
            grad_logits=grad_logits
        )

        optimizer.step()

    avg_loss = total_loss / num_samples
    avg_supcon_loss = total_supcon_loss / num_samples
    avg_ce_loss = total_ce_loss / num_samples
    accuracy = correct / num_samples

    return {
        'loss': float(avg_loss),
        'supcon_loss': float(avg_supcon_loss),
        'ce_loss': float(avg_ce_loss),
        'accuracy': float(accuracy)
    }


# ------------------------------
# 单个 fold 的训练（对齐训练脚本 + SupCon）
# ------------------------------
def train_single_fold(
    train_files: List[str],
    val_files: List[str],
    data_dir: str,
    config: Dict[str, Any],
    fold_idx: int,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Train model on one fold and return validation metrics
    """

    # 处理 augmentation / movement_features（和主训练脚本一致）
    augmentation = config.get('augmentation')
    movement_features = config.get('movement_features')
    print(f"Using augmentation: {augmentation}, movement_features: {movement_features}")

    # 创建数据集（签名对齐 DigitsStrokeDataset 在训练脚本中的用法）
    train_dataset = DigitsStrokeDataset(
        data_dir,
        seq_len=config['seq_len'],
        file_list=train_files,
        augmentation=augmentation,
        movement_features=movement_features,
        training=True
    )

    val_dataset = DigitsStrokeDataset(
        data_dir,
        seq_len=config['seq_len'],
        file_list=val_files,
        augmentation=None,  # no augmentation for val
        movement_features=movement_features,
        training=False
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=0
    )

    # 转成 numpy（一次性，和训练脚本一致）
    train_data, train_labels = numpy_from_dataloader(train_loader)
    val_data, val_labels = numpy_from_dataloader(val_loader)

    # 输入维度根据 movement_features 决定
    input_dim = get_input_dim(movement_features)

    # 是否启用 SupCon
    supcon_weight = float(config.get('supcon_weight', 0.0))
    use_supcon = supcon_weight > 0.0

    # 创建模型
    if use_supcon:
        model = TransformerSupCon(
            input_dim=input_dim,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            mlp_ratio=config.get('mlp_ratio'),
            dropout=config['dropout'],
            num_classes=config.get('num_classes', 10),
            pos_encoding=config.get('pos_encoding', 'sinusoidal'),
            projection_dim=config.get('projection_dim', 128),
            projection_hidden_dim=config.get('projection_hidden_dim', 256),
        )
        supcon_loss = SupConLoss(
            temperature=config.get('temperature', 0.07)
        )
        ce_loss = CrossEntropyLoss()
        if verbose:
            print(f"[Fold {fold_idx+1}] Using SupCon training (weight={supcon_weight})")
    else:
        model = Transformer(
            input_dim=input_dim,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            mlp_ratio=config.get('mlp_ratio'),
            dropout=config['dropout'],
            num_classes=config.get('num_classes', 10),
            pos_encoding=config.get('pos_encoding', 'sinusoidal')
        )
        if verbose:
            print(f"[Fold {fold_idx+1}] Using pure CE training")

    # 优化器（和训练脚本完全一致）
    if config['optimizer'] == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    # 学习率调度器（和训练脚本完全一致：每 epoch 调一次 step(epoch)）
    scheduler_type = config.get('scheduler', 'warmup_cosine')
    if scheduler_type == 'warmup_cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.get('warmup_epochs', 5),
            max_epochs=config['epochs'],
            base_lr=config['lr'],
            min_lr=config.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'cosine':
        eta_min = config.get('eta_min', None)
        if eta_min is None:
            eta_min = config['lr'] * 0.01
        scheduler = CosineAnnealingScheduler(
            optimizer,
            T_max=config['epochs'],
            base_lr=config['lr'],
            eta_min=eta_min
        )
    else:  # 'none'
        scheduler = None

    best_val_acc = 0.0
    best_val_loss = float('inf')

    final_val_loss = None
    final_val_acc = None

    for epoch in range(config['epochs']):
        # 每个 epoch 更新一次 LR
        if scheduler is not None:
            current_lr = scheduler.step(epoch)
        else:
            current_lr = config['lr']

        # 训练一个 epoch
        if use_supcon:
            train_stats = train_epoch_supcon(
                model=model,
                optimizer=optimizer,
                train_data=train_data,
                train_labels=train_labels,
                supcon_loss=supcon_loss,
                ce_loss=ce_loss,
                supcon_weight=supcon_weight,
                batch_size=config['batch_size']
            )
            train_loss = train_stats['loss']
            train_acc = train_stats['accuracy']
        else:
            train_loss, train_acc = train_epoch(
                model, optimizer, train_data, train_labels,
                batch_size=config['batch_size']
            )

        # 验证
        val_loss, val_acc = evaluate(
            model, val_data, val_labels,
            batch_size=config['batch_size'],
            is_supcon=use_supcon
        )

        final_val_loss = val_loss
        final_val_acc = val_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Fold {fold_idx+1} - Epoch {epoch+1}/{config['epochs']} "
                  f"(lr={current_lr:.6f}): "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return {
        'final_val_loss': float(final_val_loss),
        'final_val_acc': float(final_val_acc),
        'best_val_loss': float(best_val_loss),
        'best_val_acc': float(best_val_acc)
    }


# ------------------------------
# Grid Search with K-fold
# ------------------------------
def grid_search(
    data_dir: str,
    param_grid: Dict[str, List[Any]],
    base_config: Dict[str, Any],
    n_folds: int = 5,
    seed: int = 42,
    output_dir: str = 'grid_search_results',
    verbose: bool = True
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Perform grid search with K-fold cross validation
    """

    # 输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 根据文件名获取所有样本和 label（与训练数据命名规则一致）
    all_files = sorted(glob.glob(os.path.join(data_dir, "stroke_*_*.csv")))
    all_labels = [int(os.path.basename(f).split("_")[1]) for f in all_files]

    if verbose:
        print(f"Found {len(all_files)} samples across {len(set(all_labels))} classes")

    # K-fold 划分
    if verbose:
        print(f"\nCreating {n_folds}-fold stratified splits...")

    folds = create_k_folds(all_files, all_labels, n_folds=n_folds, seed=seed)

    # 生成所有超参组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combos = list(product(*param_values))

    # 确保 d_model 能被 nhead 整除
    valid_combos = []
    for combo in all_combos:
        cfg = dict(zip(param_names, combo))
        if 'd_model' in cfg and 'nhead' in cfg:
            if cfg['d_model'] % cfg['nhead'] == 0:
                valid_combos.append(combo)
        else:
            valid_combos.append(combo)

    if len(valid_combos) < len(all_combos):
        print(f"Note: Filtered out invalid d_model/nhead combinations")
        print(f"Valid combinations: {len(valid_combos)}")

        # 用有效组合重建 param_grid（去重）
        filtered_grid: Dict[str, set] = {k: set() for k in param_names}
        for combo in valid_combos:
            for k, v in zip(param_names, combo):
                filtered_grid[k].add(v)
        param_grid = {k: sorted(list(v)) for k, v in filtered_grid.items()}

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

    param_combinations = list(product(*param_values))

    if verbose:
        print(f"\nTesting {len(param_combinations)} hyperparameter combinations")
        print(f"Total training runs: {len(param_combinations) * n_folds}\n")

    all_results: List[Dict[str, Any]] = []

    # 遍历所有组合
    for combo_idx, param_combo in enumerate(param_combinations):
        # 合成 config（param_grid 覆盖 base_config）
        config = base_config.copy()
        for name, value in zip(param_names, param_combo):
            config[name] = value

        if verbose:
            print(f"\n{'='*80}")
            print(f"Combination {combo_idx + 1}/{len(param_combinations)}")
            print(f"{'='*80}")
            print("Parameters:")
            for name, value in zip(param_names, param_combo):
                print(f"  {name}: {value}")
            print()

        fold_results = []
        start_time = time.time()

        for fold_idx, (train_files, val_files) in enumerate(folds):
            if verbose:
                print(f"Fold {fold_idx + 1}/{n_folds} - "
                      f"Train: {len(train_files)}, Val: {len(val_files)}")

            metrics = train_single_fold(
                train_files=train_files,
                val_files=val_files,
                data_dir=data_dir,
                config=config,
                fold_idx=fold_idx,
                verbose=verbose
            )
            fold_results.append(metrics)

            if verbose:
                print(f"  Final Val Acc: {metrics['final_val_acc']:.4f}, "
                      f"Best Val Acc: {metrics['best_val_acc']:.4f}")

        elapsed_time = time.time() - start_time

        # 统计各 fold 的均值和方差
        avg_final_val_loss = np.mean([r['final_val_loss'] for r in fold_results])
        avg_final_val_acc = np.mean([r['final_val_acc'] for r in fold_results])
        avg_best_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
        avg_best_val_acc = np.mean([r['best_val_acc'] for r in fold_results])

        std_final_val_acc = np.std([r['final_val_acc'] for r in fold_results])
        std_best_val_acc = np.std([r['best_val_acc'] for r in fold_results])

        result = {
            'config': config.copy(),
            'avg_final_val_loss': float(avg_final_val_loss),
            'avg_final_val_acc': float(avg_final_val_acc),
            'avg_best_val_loss': float(avg_best_val_loss),
            'avg_best_val_acc': float(avg_best_val_acc),
            'std_final_val_acc': float(std_final_val_acc),
            'std_best_val_acc': float(std_best_val_acc),
            'fold_results': fold_results,
            'elapsed_time': float(elapsed_time)
        }

        all_results.append(result)

        if verbose:
            print(f"\n{'─'*80}")
            print(f"Average Results:")
            print(f"  Final Val Acc: {avg_final_val_acc:.4f} ± {std_final_val_acc:.4f}")
            print(f"  Best  Val Acc: {avg_best_val_acc:.4f} ± {std_best_val_acc:.4f}")
            print(f"  Time: {elapsed_time:.1f}s")

    # 找到最优配置（按 avg_best_val_acc 最大）
    best_result = max(all_results, key=lambda x: x['avg_best_val_acc'])
    best_config = best_result['config']

    if verbose:
        print(f"\n{'='*80}")
        print("BEST CONFIGURATION")
        print(f"{'='*80}")
        print(f"Best Val Acc: {best_result['avg_best_val_acc']:.4f} "
              f"± {best_result['std_best_val_acc']:.4f}")
        print("\nParameters:")
        for k, v in best_config.items():
            if k in param_names:
                print(f"  {k}: {v}")

    # 保存结果到 json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"grid_search_{timestamp}.json"

    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(x) for x in obj]
        return obj

    results_data = {
        'best_config': convert_to_serializable(best_config),
        'best_result': convert_to_serializable(best_result),
        'all_results': convert_to_serializable(all_results),
        'param_grid': convert_to_serializable(param_grid),
        'base_config': convert_to_serializable(base_config),
        'n_folds': n_folds,
        'seed': seed,
        'timestamp': timestamp
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    if verbose:
        print(f"\nResults saved to: {results_file}")

    return best_config, all_results


# ------------------------------
# CLI 入口：阶段式搜索版本 + SupCon
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Grid Search (manual backprop Transformer + SupCon) with K-Fold CV'
    )

    # 数据相关
    parser.add_argument('--data_dir', type=str, default='../../digits_3d/training_data',
                        help='Path to training data')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # 输出
    parser.add_argument('--output_dir', type=str, default='grid_search_results',
                        help='Directory to save grid search results')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    # 固定基本训练参数（和训练脚本一致）
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs per fold')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes')

    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-2,
                        help='Weight decay')
    parser.add_argument('--mlp_ratio', type=float, default=None,
                        help='MLP ratio (None = default in model)')

    # optimizer / scheduler 选项（这里只是 base_config，真正搜索范围在 param_grid 里）
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adamw'],
                        help='Default optimizer type (if not overridden by grid)')
    parser.add_argument('--scheduler', type=str, default='warmup_cosine',
                        choices=['warmup_cosine', 'cosine', 'none'],
                        help='Default lr scheduler type (if not overridden by grid)')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Warmup epochs (for warmup_cosine)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum lr (for warmup_cosine)')
    parser.add_argument('--eta_min', type=float, default=None,
                        help='Minimum lr for cosine (default: lr * 0.01)')
    parser.add_argument('--pos_encoding', type=str, default='sinusoidal',
                        choices=['sinusoidal', 'learnable', 'conditional'],
                        help='Positional encoding type')

    # SupCon 相关参数（用于 grid search）
    parser.add_argument('--supcon_weight', type=float, default=0.0,
                        help='Weight for SupCon loss (0=disable SupCon)')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for SupCon loss')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Projection head output dimension for SupCon')
    parser.add_argument('--projection_hidden_dim', type=int, default=256,
                        help='Projection head hidden dimension for SupCon')

    # Grid search 模式
    parser.add_argument('--search_mode', type=str, default='quick',
                        choices=['quick', 'standard', 'extensive', 'custom'],
                        help='Predefined search mode')

    # Custom grid search 参数 (仅在 --search_mode custom/standard/extensive 时用)
    parser.add_argument('--lr_values', type=float, nargs='+',
                        help='Learning rates to search')
    parser.add_argument('--d_model_values', type=int, nargs='+',
                        help='Model dimensions to search')
    parser.add_argument('--num_layers_values', type=int, nargs='+',
                        help='Number of layers to search')
    parser.add_argument('--dropout_values', type=float, nargs='+',
                        help='Dropout rates to search')
    parser.add_argument('--nhead_values', type=int, nargs='+',
                        help='Number of heads to search')
    parser.add_argument('--mlp_ratio_values', type=float, nargs='+',
                        help='MLP ratios to search')

    # SupCon 超参搜索
    parser.add_argument('--supcon_weight_values', type=float, nargs='+',
                        help='SupCon weights to search (include 0.0 to disable)')
    parser.add_argument('--temperature_values', type=float, nargs='+',
                        help='SupCon temperatures to search')

    # augmentation / movement_features 的 CLI 默认值（可配合 custom_values ）
    parser.add_argument('--augmentation', type=str, default=None,
                        help='Default augmentation mode (used if no augmentation_values provided)')
    parser.add_argument('--movement_features', type=str, default=None,
                        help='Default movement_features mode (used if no movement_features_values provided)')

    # 新增：augmentation / movement_features 的搜索值（custom 模式使用）
    parser.add_argument('--augmentation_values', type=str, nargs='+',
                        help='Augmentation options to search (e.g. light medium strong none)')
    parser.add_argument('--movement_features_values', type=str, nargs='+',
                        help='Movement feature modes to search (e.g. concat replace all none)')

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # base_config：不在 grid 里的东西用这里的值
    base_config: Dict[str, Any] = {
        'seq_len': args.seq_len,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'num_classes': args.num_classes,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'mlp_ratio': args.mlp_ratio,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'warmup_epochs': args.warmup_epochs,
        'min_lr': args.min_lr,
        'eta_min': args.eta_min,
        'pos_encoding': args.pos_encoding,
        # SupCon
        'supcon_weight': args.supcon_weight,
        'temperature': args.temperature,
        'projection_dim': args.projection_dim,
        'projection_hidden_dim': args.projection_hidden_dim,
    }

    # 定义要搜索的超参数空间
    if args.search_mode == 'quick':
        # quick 模式用作“全集”，阶段式搜索时按组拆开
        quick_grid: Dict[str, List[Any]] = {
            'lr': [1e-3],
            'd_model': [64],
            'nhead': [4],
            'num_layers': [2],
            'dropout': [0.1],
            'weight_decay': [5e-2],
            'optimizer': ['adamw'],
            'scheduler': ['warmup_cosine'],
            'pos_encoding': ['sinusoidal'],
            'mlp_ratio': [None],
            'augmentation': ['light'],
            'movement_features': ['concat'],
            # SupCon: 0.0 表示关闭；>0 表示启用 SupCon
            'supcon_weight': [0.1, 0.3, 0.5, 0.7],
            'temperature': [0.05, 0.07, 0.1],
            'projection_dim': [32, 64],
            'projection_hidden_dim': [32, 64],
        }
        param_grid = None  # 只是为了后面类型一致，不会用到
    else:  # custom / standard / extensive（这里统一走 custom 的逻辑）
        param_grid: Dict[str, List[Any]] = {}

        param_grid['lr'] = args.lr_values if args.lr_values else [1e-3, 5e-4, 1e-4]
        param_grid['d_model'] = args.d_model_values if args.d_model_values else [128, 256]
        param_grid['num_layers'] = args.num_layers_values if args.num_layers_values else [2, 3, 4]
        param_grid['dropout'] = args.dropout_values if args.dropout_values else [0.1, 0.2]
        param_grid['nhead'] = args.nhead_values if args.nhead_values else [4, 8]
        param_grid['mlp_ratio'] = args.mlp_ratio_values if args.mlp_ratio_values else [None, 2.0, 4.0]

        # SupCon 超参
        if args.supcon_weight_values:
            param_grid['supcon_weight'] = args.supcon_weight_values
        else:
            param_grid['supcon_weight'] = [args.supcon_weight]

        if args.temperature_values:
            param_grid['temperature'] = args.temperature_values
        else:
            param_grid['temperature'] = [args.temperature]

        # projection_dim / hidden_dim 默认不大范围搜索
        param_grid['projection_dim'] = [args.projection_dim]
        param_grid['projection_hidden_dim'] = [args.projection_hidden_dim]

        # 处理 augmentation grid（支持命令行传多值）
        if args.augmentation_values:
            aug_grid_raw = args.augmentation_values
        else:
            # 如果没显式给，就只用当前 CLI 设置的 augmentation
            aug_grid_raw = [args.augmentation]

        aug_grid: List[Any] = []
        for a in aug_grid_raw:
            if a in [None, 'none', 'None']:
                aug_grid.append(None)
            else:
                aug_grid.append(a)
        param_grid['augmentation'] = aug_grid

        # 处理 movement_features grid
        if args.movement_features_values:
            mf_grid_raw = args.movement_features_values
        else:
            mf_grid_raw = [args.movement_features]

        mf_grid: List[Any] = []
        for m in mf_grid_raw:
            if m in [None, 'none', 'None']:
                mf_grid.append(None)
            else:
                mf_grid.append(m)
        param_grid['movement_features'] = mf_grid

        # 其它保持默认或少量搜索
        param_grid['weight_decay'] = [args.weight_decay]
        param_grid['optimizer'] = ['adamw', 'sgd']
        param_grid['scheduler'] = ['warmup_cosine', 'cosine', 'none']
        param_grid['pos_encoding'] = ['sinusoidal', 'learnable', 'conditional']

    # 打印基本信息
    print(f"\n{'='*80}")
    print(f"Grid Search with {args.n_folds}-Fold Cross Validation")
    print(f"{'='*80}")
    print(f"Search mode: {args.search_mode}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs per fold: {args.epochs}")
    print(f"Random seed: {args.seed}")
    print()

    # --------------------------
    # quick 模式：多阶段（stage-wise）搜索
    # --------------------------
    if args.search_mode == 'quick':
        # 分阶段的超参数组：
        # Stage 1: 结构相关（模型容量）
        # Stage 2: 数据表示 & SupCon
        # Stage 3: 优化相关（lr / weight_decay / scheduler / optimizer）
        # stage_groups: List[List[str]] = [
        #     ['d_model', 'nhead', 'num_layers', 'dropout'],
        #     ['augmentation', 'movement_features', 'mlp_ratio', 'pos_encoding',
        #      'supcon_weight', 'temperature', 'projection_dim', 'projection_hidden_dim'],
        #     ['lr', 'weight_decay', 'scheduler', 'optimizer'],
        # ]
        stage_groups: List[List[str]] = [
            ['d_model', 'nhead', 'num_layers', 'dropout', 'augmentation', 'movement_features', 'mlp_ratio', 'pos_encoding',
             'supcon_weight', 'temperature', 'projection_dim', 'projection_hidden_dim',
             'lr', 'weight_decay', 'scheduler', 'optimizer'],
        ]

        best_config_overall: Dict[str, Any] = base_config.copy()
        all_results = None

        for stage_idx, stage_keys in enumerate(stage_groups, start=1):
            # 构造当前阶段的 param_grid：
            #   - 当前阶段的 key 用 quick_grid 中的全部候选值
            #   - 其他 key 固定为“当前最优配置里的值”（如果还没有，就用 quick_grid 的第一个值）
            stage_param_grid: Dict[str, List[Any]] = {}
            for k, vals in quick_grid.items():
                if k in stage_keys:
                    stage_param_grid[k] = vals
                else:
                    if k in best_config_overall:
                        stage_param_grid[k] = [best_config_overall[k]]
                    else:
                        stage_param_grid[k] = [vals[0]]

            print(f"\n{'#' * 80}")
            print(f"Stage {stage_idx}: searching hyperparameters: {stage_keys}")
            print(f"{'#' * 80}\n")

            best_config_stage, all_results = grid_search(
                data_dir=args.data_dir,
                param_grid=stage_param_grid,
                base_config=base_config,
                n_folds=args.n_folds,
                seed=args.seed,
                output_dir=args.output_dir,
                verbose=not args.quiet
            )

            # 把这一阶段找到的最优配置合并进 overall
            best_config_overall.update(best_config_stage)

        best_config = best_config_overall

    # --------------------------
    # 其它模式：一次性 grid search
    # --------------------------
    else:
        best_config, all_results = grid_search(
            data_dir=args.data_dir,
            param_grid=param_grid,
            base_config=base_config,
            n_folds=args.n_folds,
            seed=args.seed,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )

    print("\nGrid search completed!")
    print("Best config:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()