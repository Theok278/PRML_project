import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


class CPE(nn.Module):
    """Conditional Position Encoding using 3D convolution"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)

    def forward(self, x):
        # x: (N, C)
        # Reshape for 1D conv: (1, C, N)
        x_conv = x.t().unsqueeze(0)  # (1, C, N)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.squeeze(0).t()  # (N, C)
        return x + x_conv


class Attention(nn.Module):
    """Multi-head Self-Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        N, C = x.shape
        qkv = self.qkv(x).reshape(N, 3, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP with GELU activation"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer Block with CPE"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.cpe = CPE(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                             attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = self.cpe(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DownSample(nn.Module):
    """Down sampling layer between encoder stages"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # Simple linear projection for down sampling
        # In real SONATA, this might use grid pooling
        x = self.linear(x)
        x = self.norm(x)
        return x


class SONATABackbone(nn.Module):
    """SONATA Backbone matching pretrained weights structure"""
    def __init__(
        self,
        in_channels=9,  # SONATA uses 9 channels (xyz + rgb + normalized_xyz)
        embed_dim=48,
        depths=[3, 3, 3, 12, 3],  # enc0-enc4
        channels=[48, 96, 192, 384, 512],
        num_heads=[2, 4, 8, 16, 32],
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.
    ):
        super().__init__()

        # Embedding stem
        self.embedding = nn.ModuleDict({
            'stem': nn.ModuleDict({
                'linear': nn.Linear(in_channels, embed_dim, bias=False),
                'norm': nn.LayerNorm(embed_dim)
            })
        })

        # Build encoder stages
        self.enc = nn.ModuleDict()
        for stage_idx, (depth, dim, num_head) in enumerate(zip(depths, channels, num_heads)):
            stage_name = f'enc{stage_idx}'
            stage_dict = nn.ModuleDict()

            # Down sampling (except for first stage)
            if stage_idx > 0:
                stage_dict['down'] = DownSample(channels[stage_idx-1], dim)

            # Transformer blocks
            for block_idx in range(depth):
                block_name = f'block{block_idx}'
                stage_dict[block_name] = Block(
                    dim=dim,
                    num_heads=num_head,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate
                )

            self.enc[stage_name] = stage_dict

        self.num_features = channels[-1]

    def forward(self, coord, feat, offset=None):
        """
        Args:
            coord: (N, 3) point coordinates
            feat: (N, C) point features (should be 9 channels for SONATA)
            offset: (B,) batch offsets
        Returns:
            features: (N, final_dim) point-wise features
            global_feat: (B, final_dim) global features
        """
        # Embedding
        x = self.embedding['stem']['linear'](feat)
        x = self.embedding['stem']['norm'](x)

        # Encoder stages
        for stage_idx, stage_name in enumerate(['enc0', 'enc1', 'enc2', 'enc3', 'enc4']):
            stage = self.enc[stage_name]

            # Down sampling
            if stage_idx > 0:
                x = stage['down'](x)

            # Transformer blocks
            for key in stage.keys():
                if key.startswith('block'):
                    x = stage[key](x)

        # Global pooling per batch
        if offset is not None:
            batch_size = len(offset)
            global_feats = []
            for i in range(batch_size):
                start_idx = offset[i]
                end_idx = offset[i + 1] if i + 1 < batch_size else x.shape[0]
                batch_feat = x[start_idx:end_idx]
                global_feat = batch_feat.max(dim=0)[0]  # Max pooling
                global_feats.append(global_feat)
            global_feat = torch.stack(global_feats, dim=0)  # (B, final_dim)
        else:
            global_feat = x.max(dim=0, keepdim=True)[0]  # (1, final_dim)

        return x, global_feat


class ClassificationHead(nn.Module):
    """Classification head for transfer learning"""
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)


class SONATAClassifier(nn.Module):
    """SONATA model with classification head for transfer learning"""
    def __init__(
        self,
        num_classes=10,
        freeze_encoder=True,
        dropout=0.5
    ):
        super().__init__()

        # SONATA backbone matching pretrained weights
        self.backbone = SONATABackbone(
            in_channels=9,  # SONATA expects 9 channels
            embed_dim=48,
            depths=[3, 3, 3, 12, 3],
            channels=[48, 96, 192, 384, 512],
            num_heads=[2, 4, 8, 16, 32],
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.
        )

        # Classification head
        self.head = ClassificationHead(512, num_classes, dropout)

        # Freeze encoder for transfer learning
        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self):
        """Freeze encoder parameters for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Encoder frozen - only training classification head")

    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Encoder unfrozen - training full model")

    def forward(self, data_dict):
        """
        Args:
            data_dict: {
                'coord': (N, 3) or (B*N, 3)
                'feat': (N, C) or (B*N, C)  - should be 9 channels
                'offset': (B,) batch offsets
            }
        Returns:
            logits: (B, num_classes)
        """
        coord = data_dict['coord']
        feat = data_dict['feat']
        offset = data_dict.get('offset', None)

        # Get features from backbone
        point_feat, global_feat = self.backbone(coord, feat, offset)

        # Classification
        logits = self.head(global_feat)

        return logits

    def load_pretrained(self, checkpoint_path, strict=False):
        """Load pretrained SONATA weights"""
        print(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Extract student backbone weights and remap keys
        backbone_state_dict = {}
        for k, v in state_dict.items():
            # We want keys like: module.student.backbone.xxx
            if 'student.backbone' in k:
                # Remove 'module.student.backbone.' prefix
                new_k = k.replace('module.student.backbone.', '')
                backbone_state_dict[new_k] = v

        # Load weights into backbone
        msg = self.backbone.load_state_dict(backbone_state_dict, strict=strict)
        print(f"Loaded backbone weights:")
        print(f"  Missing keys: {len(msg.missing_keys)}")
        print(f"  Unexpected keys: {len(msg.unexpected_keys)}")

        if len(msg.missing_keys) > 0:
            print(f"  First few missing: {msg.missing_keys[:5]}")
        if len(msg.unexpected_keys) > 0:
            print(f"  First few unexpected: {msg.unexpected_keys[:5]}")

        return msg


def build_sonata_classifier(num_classes=10, pretrained_path=None, freeze_encoder=True):
    """Build SONATA classifier model"""
    model = SONATAClassifier(
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        dropout=0.5
    )

    if pretrained_path:
        model.load_pretrained(pretrained_path, strict=False)

    return model
