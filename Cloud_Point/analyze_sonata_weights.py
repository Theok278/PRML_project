#!/usr/bin/env python3
"""分析 SONATA 预训练权重的详细结构"""

import torch
from collections import defaultdict

# 加载预训练权重
checkpoint_path = './sonata/pretrain-sonata-v1m1-0-base.pth'
print(f"Loading {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# 获取 state_dict
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint

print(f"\n总共 {len(state_dict)} 个参数")

# 分析层结构
print("\n" + "="*80)
print("层结构分析:")
print("="*80)

# 按前缀分组
prefix_groups = defaultdict(list)
for key in state_dict.keys():
    parts = key.split('.')
    if len(parts) >= 3:
        prefix = '.'.join(parts[:3])
        prefix_groups[prefix].append(key)

# 打印每个组的信息
for prefix in sorted(prefix_groups.keys()):
    keys = prefix_groups[prefix]
    print(f"\n{prefix}:")
    print(f"  数量: {len(keys)} 个参数")
    # 显示前3个键
    for k in keys[:3]:
        shape = state_dict[k].shape if hasattr(state_dict[k], 'shape') else 'N/A'
        print(f"    {k}: {shape}")
    if len(keys) > 3:
        print(f"    ... 还有 {len(keys)-3} 个")

# 分析 embedding stem
print("\n" + "="*80)
print("Embedding 层:")
print("="*80)
for key in sorted(state_dict.keys()):
    if 'embedding' in key:
        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
        print(f"{key}: {shape}")

# 分析 enc0
print("\n" + "="*80)
print("enc0 结构 (第一阶段):")
print("="*80)
enc0_keys = [k for k in state_dict.keys() if 'enc.enc0' in k]
enc0_blocks = set()
for key in enc0_keys:
    parts = key.split('.')
    for i, part in enumerate(parts):
        if part.startswith('block'):
            enc0_blocks.add(part)

print(f"enc0 有 {len(enc0_blocks)} 个 blocks: {sorted(enc0_blocks)}")

# 显示一个 block 的结构
print("\nenc0.block0 的层:")
for key in sorted([k for k in enc0_keys if 'block0' in k])[:10]:
    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
    print(f"  {key}: {shape}")

# 统计所有 enc 阶段
print("\n" + "="*80)
print("所有 enc 阶段:")
print("="*80)
for i in range(5):
    enc_keys = [k for k in state_dict.keys() if f'enc.enc{i}' in k]
    if enc_keys:
        blocks = set()
        for key in enc_keys:
            parts = key.split('.')
            for j, part in enumerate(parts):
                if part.startswith('block'):
                    blocks.add(part)
        print(f"enc{i}: {len(enc_keys)} 个参数, {len(blocks)} 个 blocks")

# 检查 down 采样层
print("\n" + "="*80)
print("Down 采样层:")
print("="*80)
for key in sorted(state_dict.keys()):
    if '.down.' in key:
        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
        print(f"{key}: {shape}")
