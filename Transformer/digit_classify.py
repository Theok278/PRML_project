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

    # 这两项完全从 checkpoint 里读
    seq_len = args.get('seq_len', 128)
    movement_features = args.get('movement_features', None)

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
        self.input_dim = self.config["input_dim"]

    def preprocess(self, testdata):
        from dataset import resample_points, pretreat_points
        import numpy as np

        # ==== 下面这段直接沿用你原来的逻辑 ====
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

        testdata = testdata.astype(np.float32)

        # 重采样到 seq_len（从 checkpoint 里的 args 读出来的）
        resampled = resample_points(testdata, self.seq_len)

        # 归一化（与训练时保持一致）
        resampled = pretreat_points(resampled, normalization=True)

        # movement_features 也从 args 决定
        if self.movement_features == 'concat':
            velocity = np.diff(resampled, axis=0, prepend=resampled[0:1])
            resampled = np.concatenate([resampled, velocity], axis=1)

        elif self.movement_features == 'replace':
            velocity = np.diff(resampled, axis=0, prepend=resampled[0:1])
            resampled = velocity

        elif self.movement_features == 'all':
            velocity = np.diff(resampled, axis=0, prepend=resampled[0:1])
            acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
            resampled = np.concatenate([resampled, velocity, acceleration], axis=1)

        elif self.movement_features == 'velocity_acceleration':
            velocity = np.diff(resampled, axis=0, prepend=resampled[0:1])
            acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
            resampled = np.concatenate([velocity, acceleration], axis=1)

        elif self.movement_features == 'acceleration_only':
            velocity = np.diff(resampled, axis=0, prepend=resampled[0:1])
            acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
            resampled = acceleration

        batch_data = resampled[np.newaxis, ...]
        return batch_data

    def classify(self, testdata) -> int:
        batch_data = self.preprocess(testdata)

        if self.model_type == "supcon":
            output = self.model.forward(batch_data, training=False, return_embeddings=False)
            logits = output
        else:
            logits = self.model.forward(batch_data, training=False)

        predicted_class = int(np.argmax(logits[0]))
        return predicted_class

    def classify_with_confidence(self, testdata):
        batch_data = self.preprocess(testdata)

        if self.model_type == "supcon":
            output = self.model.forward(batch_data, training=False, return_embeddings=False)
            logits = output
        else:
            logits = self.model.forward(batch_data, training=False)

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