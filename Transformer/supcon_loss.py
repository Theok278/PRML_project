"""
Supervised Contrastive Loss (SupCon) with manual backpropagation

Based on "Supervised Contrastive Learning" (Khosla et al., 2020)
https://arxiv.org/abs/2004.11362

Reference implementation:
https://github.com/google-research/google-research/tree/master/supcon
"""

import numpy as np


class SupConLoss:
    """
    Supervised Contrastive Learning Loss

    For each anchor, pull positive samples (same class) closer
    and push negative samples (different class) further away.

    Loss formula:
        L = -1/|P(i)| * sum_{p in P(i)} log[ exp(z_i · z_p / τ) / sum_{a in A(i)} exp(z_i · z_a / τ) ]

    where:
        - P(i): set of indices of positives for anchor i (same class, excluding i)
        - A(i): set of all indices excluding i
        - τ: temperature parameter
        - z: L2-normalized embeddings
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
            base_temperature: Base temperature (for normalization)
        """
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.cache = {}

    def forward(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute supervised contrastive loss

        Args:
            features: (batch_size, feat_dim) - embeddings (NOT normalized)
            labels: (batch_size,) - class labels

        Returns:
            loss: scalar loss value
        """
        batch_size = features.shape[0]

        if batch_size < 2:
            raise ValueError(f"Batch size must be at least 2, got {batch_size}")

        # L2 normalize features
        features_norm = np.linalg.norm(features, axis=1, keepdims=True)
        features_normalized = features / (features_norm + 1e-8)

        # Compute similarity matrix: (batch, batch)
        # similarity[i, j] = cos_sim(features[i], features[j])
        similarity_matrix = np.dot(features_normalized, features_normalized.T)

        # Create mask for positive pairs (same class, excluding diagonal)
        labels = labels.reshape(-1, 1)  # (batch, 1)
        mask_positive = (labels == labels.T).astype(np.float32)  # (batch, batch)
        mask_positive[np.arange(batch_size), np.arange(batch_size)] = 0  # Exclude self

        # Create mask for valid pairs (all except diagonal)
        mask_valid = np.ones((batch_size, batch_size), dtype=np.float32)
        mask_valid[np.arange(batch_size), np.arange(batch_size)] = 0

        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature

        # For numerical stability, subtract max
        logits_max = np.max(similarity_matrix, axis=1, keepdims=True)
        logits = similarity_matrix - logits_max

        # Compute exp
        exp_logits = np.exp(logits) * mask_valid  # Exclude diagonal

        # Sum of exp over all valid pairs (denominator)
        sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)  # (batch, 1)

        # Log probabilities
        log_prob = logits - np.log(sum_exp_logits + 1e-8)

        # Mean of log-likelihood over positive pairs
        # For each sample, average over all its positive pairs
        num_positives = np.sum(mask_positive, axis=1)  # (batch,)

        # Avoid division by zero (samples with no positives)
        valid_samples = num_positives > 0

        if not np.any(valid_samples):
            # No valid positive pairs in batch
            print("Warning: No positive pairs found in batch")
            return 0.0

        # Sum log probabilities over positive pairs, then average
        mean_log_prob_pos = np.sum(mask_positive * log_prob, axis=1) / (num_positives + 1e-8)

        # Only average over samples that have positives
        loss = -np.mean(mean_log_prob_pos[valid_samples])

        # Scale by temperature
        loss = loss * (self.temperature / self.base_temperature)

        # Cache for backward
        self.cache = {
            'features': features,
            'features_normalized': features_normalized,
            'features_norm': features_norm,
            'similarity_matrix': similarity_matrix,
            'exp_logits': exp_logits,
            'sum_exp_logits': sum_exp_logits,
            'log_prob': log_prob,
            'mask_positive': mask_positive,
            'mask_valid': mask_valid,
            'num_positives': num_positives,
            'valid_samples': valid_samples,
            'labels': labels,
            'batch_size': batch_size
        }

        return loss

    def backward(self) -> np.ndarray:
        """
        Backward pass for supervised contrastive loss

        Returns:
            grad_features: (batch_size, feat_dim) - gradient w.r.t. input features
        """
        # Retrieve cached values
        features = self.cache['features']
        features_normalized = self.cache['features_normalized']
        features_norm = self.cache['features_norm']
        exp_logits = self.cache['exp_logits']
        sum_exp_logits = self.cache['sum_exp_logits']
        mask_positive = self.cache['mask_positive']
        mask_valid = self.cache['mask_valid']
        num_positives = self.cache['num_positives']
        valid_samples = self.cache['valid_samples']
        batch_size = self.cache['batch_size']

        # Compute gradient of log_prob w.r.t. similarity matrix
        # d(log_prob)/d(sim) = 1_{positive} - exp(sim) / sum_exp
        prob_matrix = exp_logits / (sum_exp_logits + 1e-8)  # (batch, batch)

        # Weight by mask_positive and normalize by num_positives
        grad_log_prob = mask_positive / (num_positives[:, None] + 1e-8) - prob_matrix * mask_valid

        # Only apply gradient for valid samples
        grad_log_prob = grad_log_prob * valid_samples[:, None]

        # Average over batch and scale by temperature
        num_valid = np.sum(valid_samples)
        grad_log_prob = -grad_log_prob / (num_valid + 1e-8)
        grad_log_prob = grad_log_prob * (self.temperature / self.base_temperature)

        # Gradient w.r.t. similarity matrix (before temperature scaling)
        grad_similarity = grad_log_prob / self.temperature

        # Gradient w.r.t. normalized features
        # similarity[i,j] = features_normalized[i] · features_normalized[j]
        # d(similarity)/d(features_normalized[i]) = sum_j grad_similarity[i,j] * features_normalized[j]
        #                                          + sum_k grad_similarity[k,i] * features_normalized[k]
        grad_features_normalized = (
            np.dot(grad_similarity, features_normalized) +
            np.dot(grad_similarity.T, features_normalized)
        )

        # Gradient through L2 normalization
        # If y = x / ||x||, then dy/dx = (I - yy^T) / ||x||
        dot_product = np.sum(grad_features_normalized * features_normalized, axis=1, keepdims=True)
        grad_features = (grad_features_normalized - dot_product * features_normalized) / (features_norm + 1e-8)

        return grad_features


class CombinedLoss:
    """
    Combined Loss: SupCon + CrossEntropy

    Total Loss = λ * L_SupCon + (1-λ) * L_CE

    Where:
        - L_SupCon: Supervised contrastive loss (learns representation)
        - L_CE: Cross-entropy loss (learns decision boundary)
        - λ: Weight balancing the two losses
    """

    def __init__(self, supcon_loss: SupConLoss, ce_loss,
                 supcon_weight: float = 0.5):
        """
        Args:
            supcon_loss: SupConLoss instance
            ce_loss: CrossEntropyLoss instance
            supcon_weight: Weight for SupCon loss (0.0 = CE only, 1.0 = SupCon only)
        """
        self.supcon_loss = supcon_loss
        self.ce_loss = ce_loss
        self.supcon_weight = supcon_weight
        self.ce_weight = 1.0 - supcon_weight

        self.cache = {}

    def forward(self, embeddings: np.ndarray, logits: np.ndarray,
                labels: np.ndarray) -> float:
        """
        Compute combined loss

        Args:
            embeddings: (batch_size, embed_dim) - features for contrastive loss
            logits: (batch_size, num_classes) - logits for classification
            labels: (batch_size,) - ground truth labels

        Returns:
            total_loss: scalar
        """
        # Compute individual losses
        supcon_loss_val = self.supcon_loss.forward(embeddings, labels)
        ce_loss_val = self.ce_loss.forward(logits, labels)

        # Combined loss
        total_loss = (self.supcon_weight * supcon_loss_val +
                     self.ce_weight * ce_loss_val)

        # Cache for backward
        self.cache = {
            'supcon_loss': supcon_loss_val,
            'ce_loss': ce_loss_val
        }

        return total_loss

    def backward(self) -> tuple:
        """
        Backward pass for combined loss

        Returns:
            (grad_embeddings, grad_logits)
        """
        # Backprop through both losses
        grad_embeddings = self.supcon_loss.backward() * self.supcon_weight
        grad_logits = self.ce_loss.backward() * self.ce_weight

        return grad_embeddings, grad_logits

    def get_loss_components(self) -> dict:
        """Return individual loss components for logging"""
        return {
            'supcon_loss': self.cache['supcon_loss'],
            'ce_loss': self.cache['ce_loss'],
            'supcon_weight': self.supcon_weight,
            'ce_weight': self.ce_weight
        }
