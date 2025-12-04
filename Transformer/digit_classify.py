import numpy as np
from pathlib import Path
import json

# fixed checkpoint path
ckpt_dir = Path(__file__).parent / "checkpoint"
npz_files = list(ckpt_dir.glob("*.npz"))

if len(npz_files) == 0:
    raise FileNotFoundError("No .npz checkpoint found in ./checkpoint/")
elif len(npz_files) > 1:
    raise RuntimeError("Multiple .npz files found; please keep only one.")
else:
    CHECKPOINT_PATH = npz_files[0]

# 全局分类器实例（缓存模型，保证只加载一次）
_classifier = None

def load_parameters_strict(model, checkpoint, strict=True):
    model_params = list(model.parameters())
    total = len(model_params)

    loaded = 0
    missing = 0
    mismatch = 0

    for i, param in enumerate(model_params):
        key = f"param_{i}"

        if key not in checkpoint:
            missing += 1
            if strict:
                raise ValueError(f"[STRICT LOAD ERROR] Missing parameter: {key}")
            continue

        ckpt_param = checkpoint[key]

        # shape 不一致
        if ckpt_param.shape != param.data.shape:
            mismatch += 1
            if strict:
                raise ValueError(
                    f"[STRICT LOAD ERROR] Shape mismatch for {key}: "
                    f"checkpoint {ckpt_param.shape}, model {param.data.shape}"
                )
            continue

        # load
        param.data[:] = ckpt_param
        loaded += 1

    print(f"\n=== Parameter Load Summary ===")
    print(f"  Loaded        : {loaded}/{total}")
    print(f"  Missing       : {missing}")
    print(f"  Shape mismatch: {mismatch}")
    print(f"  Strict mode   : {strict}")
    print("================================\n")

    return loaded, missing, mismatch

def load_model_from_checkpoint(checkpoint_path: Path):
    """
    从 checkpoint 自动加载模型及配置（完全依赖 args）
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = np.load(checkpoint_path, allow_pickle=True)

    if 'args' not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain 'args' key. "
            "Please ensure the checkpoint was saved with training arguments."
        )

    # 从 args 中读取配置（包含 seq_len / movement_features）
    args = json.loads(str(checkpoint['args']))

    print("  Loaded config from checkpoint")

    # 基本 Transformer 配置
    d_model = args.get('d_model', 128)
    nhead = args.get('nhead', 8)
    num_layers = args.get('num_layers', 4)
    dropout = args.get('dropout', 0.1)
    mlp_ratio = args.get('mlp_ratio', None)
    pos_encoding = args.get('pos_encoding', 'sinusoidal')

    # 这些项完全从 checkpoint 里读
    seq_len = args.get('seq_len', 128)
    movement_features = args.get('movement_features', None)
    use_resample = args.get('use_resample', False)
    resample_method = args.get('resample_method', 'arclength')

    # 判断是否为 SupCon 模型
    is_supcon = 'supcon_weight' in args or 'projection_dim' in args
    supcon_weight = args.get('supcon_weight', 0.5)
    temperature = args.get('temperature', 0.07)
    projection_dim = args.get('projection_dim', 128)
    projection_hidden_dim = args.get('projection_hidden_dim', 256)

    print("Model configuration:")
    print(f"  Type: {'SupCon' if is_supcon else 'Standard'} Transformer")
    print(f"  d_model: {d_model}")
    print(f"  nhead: {nhead}")
    print(f"  num_layers: {num_layers}")
    print(f"  dropout: {dropout}")
    print(f"  pos_encoding: {pos_encoding}")
    print(f"  seq_len: {seq_len}")
    print(f"  movement_features: {movement_features}")
    print(f"  use_resample: {use_resample}")
    if use_resample:
        print(f"  resample_method: {resample_method}")
    if is_supcon:
        print(f"  supcon_weight: {supcon_weight}")
        print(f"  temperature: {temperature}")
        print(f"  projection_dim: {projection_dim}")

    # 构建模型
    try:
        from augmentation import get_input_dim
        input_dim = get_input_dim(movement_features)

        if is_supcon:
            from model_supcon import TransformerSupCon
            model = TransformerSupCon(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                num_classes=10,
                pos_encoding=pos_encoding,
                projection_dim=projection_dim,
                projection_hidden_dim=projection_hidden_dim
            )
            model_type = 'supcon'
            print("  Loaded SupCon model")
        else:
            from model import Transformer
            model = Transformer(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                num_classes=10,
                pos_encoding=pos_encoding
            )
            model_type = 'standard'
            print("  Loaded standard model")

    except ImportError as e:
        raise ImportError(
            f"Failed to import model: {e}\n"
            "Make sure model.py / model_supcon.py are in the same directory"
        )

    # 加载权重
    loaded_count = load_parameters_strict(model, checkpoint, strict=True)
    print(f"Model parameters loaded: {loaded_count}")
    model.eval()

    config = {
        "seq_len": seq_len,
        "movement_features": movement_features,
        "use_resample": use_resample,
        "resample_method": resample_method,
        "input_dim": input_dim,
        "d_model": d_model,
        "model_type": model_type,
    }

    print("Model loaded successfully!\n")
    return model, model_type, config


class DigitClassifier:
    """
    包一层，负责：
    - 调用 load_model_from_checkpoint 从固定路径加载模型
    - 保存 seq_len / movement_features 用于预处理
    """

    def __init__(self):
        self.model, self.model_type, self.config = load_model_from_checkpoint(
            CHECKPOINT_PATH
        )
        self.seq_len = self.config["seq_len"]
        self.movement_features = self.config["movement_features"]
        self.use_resample = self.config["use_resample"]
        self.resample_method = self.config["resample_method"]
        self.input_dim = self.config["input_dim"]

    def preprocess(self, testdata):
        from dataset import (
            GLOBAL_MEAN_3, GLOBAL_STD_3,
            GLOBAL_MEAN_6, GLOBAL_STD_6,
            GLOBAL_MEAN_9, GLOBAL_STD_9,
            GLOBAL_MEAN_12, GLOBAL_STD_12,
            resample_points
        )
        from augmentation import MovementFeatureExtractor, get_input_dim
        import numpy as np

        # 输入数据验证和转换
        if isinstance(testdata, (str, Path)):
            testdata = np.loadtxt(testdata, delimiter=",")
        elif isinstance(testdata, (list, tuple)):
            testdata = np.array(testdata)
        elif not isinstance(testdata, np.ndarray):
            try:
                testdata = np.array(testdata)
            except Exception as e:
                raise TypeError(
                    f"testdata must be numpy array, list, tuple, or file path. "
                    f"Got {type(testdata)}. Error: {e}"
                )

        if testdata.ndim != 2:
            raise ValueError(
                f"testdata must be a 2D array (N x 3). Got shape {testdata.shape}"
            )
        if testdata.shape[1] != 3:
            raise ValueError(
                f"testdata must have 3 columns (x, y, z). Got shape {testdata.shape}"
            )
        if testdata.shape[0] < 2:
            raise ValueError(
                f"testdata must have at least 2 points. Got {testdata.shape[0]} points"
            )

        pts = testdata.astype(np.float32)

        # 1. 重采样（如果训练时使用了resample）⭐ 关键：必须与训练时一致
        if self.use_resample:
            pts = resample_points(pts, self.seq_len)

        # 2. 提取运动特征（使用与训练时一致的 MovementFeatureExtractor）
        if self.movement_features is not None:
            if self.movement_features not in ['none', 'cat_move', 'cat_dir', 'all']:
                raise ValueError(
                    f"Unknown movement_features: {self.movement_features}. "
                    f"Must be one of: 'none', 'cat_move', 'cat_dir', 'all'"
                )
            feature_extractor = MovementFeatureExtractor(mode=self.movement_features)
            features = feature_extractor(pts)  # (N, feat_dim)
        else:
            # 没有特征提取，直接使用原始xyz
            features = pts  # (N, 3)

        # 3. 归一化（使用全局统计量）
        if self.movement_features == 'all':
            features = (features - GLOBAL_MEAN_12) / GLOBAL_STD_12
        elif self.movement_features == 'cat_dir':
            features = (features - GLOBAL_MEAN_9) / GLOBAL_STD_9
        elif self.movement_features == 'cat_move':
            features = (features - GLOBAL_MEAN_6) / GLOBAL_STD_6
        elif self.movement_features == 'none':
            features = (features - GLOBAL_MEAN_3) / GLOBAL_STD_3
        # else: 如果movement_features=None，不进行归一化

        # 4. Padding或截断到seq_len（如果没有使用resample）
        if self.use_resample:
            # 已经重采样到seq_len，所有位置都有效
            T = features.shape[0]
            if T != self.seq_len:
                # 理论上不该发生，但为了稳定性
                if T > self.seq_len:
                    features = features[:self.seq_len]
                    mask = np.ones(self.seq_len, dtype=bool)
                else:
                    pad_len = self.seq_len - T
                    features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
                    mask = np.concatenate([np.ones(T, dtype=bool),
                                          np.zeros(pad_len, dtype=bool)])
            else:
                mask = np.ones(self.seq_len, dtype=bool)
        else:
            # 使用padding/truncation
            actual_len = features.shape[0]
            if actual_len > self.seq_len:
                # 截断
                features = features[:self.seq_len]
                mask = np.ones(self.seq_len, dtype=bool)
            elif actual_len < self.seq_len:
                # Padding
                pad_len = self.seq_len - actual_len
                features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
                mask = np.concatenate([np.ones(actual_len, dtype=bool),
                                      np.zeros(pad_len, dtype=bool)])
            else:
                mask = np.ones(self.seq_len, dtype=bool)

        # 5. 转换为batch format
        batch_data = features[np.newaxis, ...]  # (1, seq_len, feat_dim)
        batch_mask = mask[np.newaxis, ...]      # (1, seq_len)

        return batch_data, batch_mask

    def classify(self, testdata) -> int:
        batch_data, batch_mask = self.preprocess(testdata)

        if self.model_type == "supcon":
            output = self.model.forward(batch_data, mask=batch_mask, training=False, return_embeddings=False)
            logits = output
        else:
            logits = self.model.forward(batch_data, mask=batch_mask, training=False)

        predicted_class = int(np.argmax(logits[0]))
        return predicted_class

    def classify_with_confidence(self, testdata):
        batch_data, batch_mask = self.preprocess(testdata)

        if self.model_type == "supcon":
            output = self.model.forward(batch_data, mask=batch_mask, training=False, return_embeddings=False)
            logits = output
        else:
            logits = self.model.forward(batch_data, mask=batch_mask, training=False)

        exp_logits = np.exp(logits[0] - np.max(logits[0]))
        probabilities = exp_logits / np.sum(exp_logits)

        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        return predicted_class, confidence, probabilities


def digit_classify(testdata):
    """
    对单条 3D 轨迹进行分类：
        C = digit_classify(testdata)

    其它超参数（seq_len / movement_features / 模型结构等）
    全部从 CHECKPOINT_PATH 对应权重里的 args 自动读取。
    """
    global _classifier
    if _classifier is None:
        _classifier = DigitClassifier()
    return _classifier.classify(testdata)


def reset_classifier():
    """
    手动清空全局模型缓存（比如想强制重新加载权重时用）
    """
    global _classifier
    _classifier = None