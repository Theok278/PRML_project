import numpy as np


class WarmupCosineScheduler:
    """Warmup + Cosine Annealing Learning Rate Scheduler

    Learning rate schedule:
    1. Linear warmup from 0 to base_lr over warmup_epochs
    2. Cosine annealing from base_lr to min_lr over remaining epochs
    """

    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int,
                 base_lr: float, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self, epoch: int = None):
        """update learning rate for the given epoch"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        lr = self.get_lr()
        self.optimizer.set_lr(lr)
        return lr

    def get_lr(self) -> float:
        """calculate current learning rate"""
        epoch = self.current_epoch

        # warmup phase
        if epoch < self.warmup_epochs:
            # linear warmup from 0 to base_lr
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # cosine annealing phase
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        return lr


class CosineAnnealingScheduler:
    """pure cosine annealing learning rate scheduler"""

    def __init__(self, optimizer, T_max: int, base_lr: float, eta_min: float = 0.0):
        """
        Args:
            optimizer: Optimizer to adjust learning rate
            T_max: Total number of epochs (one cosine cycle)
            base_lr: Initial learning rate
            eta_min: Minimum learning rate (default: 0)
        """
        self.optimizer = optimizer
        self.T_max = T_max
        self.base_lr = base_lr
        self.eta_min = eta_min
        self.current_epoch = 0

    def step(self, epoch: int = None):
        """update learning rate for the given epoch"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        lr = self.get_lr()
        self.optimizer.set_lr(lr)
        return lr

    def get_lr(self) -> float:
        """calculate current learning rate using cosine annealing"""
        # cosine annealing: lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * t / T)) / 2
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             0.5 * (1 + np.cos(np.pi * self.current_epoch / self.T_max))
        return lr
