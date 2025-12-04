import numpy as np
from pathlib import Path
import json

# Fixed checkpoint path
ckpt_dir = Path(__file__).parent / "checkpoint"
npz_files = list(ckpt_dir.glob("*.npz"))

if len(npz_files) == 0:
    raise FileNotFoundError("No .npz checkpoint found in ./checkpoint/")
elif len(npz_files) > 1:
    raise RuntimeError("Multiple .npz files found; please keep only one.")
else:
    CHECKPOINT_PATH = npz_files[0]

# Global classifier instance, lazy loaded
_classifier = None

def load_parameters_strict(model, checkpoint, strict=True):
    """ 
    load model parameters from checkpoint with strict checking.
    Because we don't have parameter names, we match parameters by order.
    """
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

        # shape mismatch
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
    Load Transformer model from checkpoint .npz file.
    Automatically reads model config from saved 'args' in checkpoint.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = np.load(checkpoint_path, allow_pickle=True)

    if 'args' not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain 'args' key. "
            "Please ensure the checkpoint was saved with training arguments."
        )

    # Load config from args (including seq_len / movement_features)
    args = json.loads(str(checkpoint['args']))

    print("  Loaded config from checkpoint")

    # Basic Transformer config
    d_model = args.get('d_model', 128)
    nhead = args.get('nhead', 8)
    num_layers = args.get('num_layers', 4)
    dropout = args.get('dropout', 0.1)
    mlp_ratio = args.get('mlp_ratio', None)
    pos_encoding = args.get('pos_encoding', 'sinusoidal')

    # These items are fully read from the checkpoint
    seq_len = args.get('seq_len', 128)
    movement_features = args.get('movement_features', None)
    use_resample = args.get('use_resample', False)
    resample_method = args.get('resample_method', 'arclength')

    # Determine if this is a SupCon model
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

    # Build model
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

    # Load model parameters from checkpoint with strict checking
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
    DigitClassifier class for classifying single 3D digit trajectories.

    Usage:
        classifier = DigitClassifier()
        class_label = classifier.classify(testdata)
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

        # Input data validation and conversion
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

        # 1. Resample
        if self.use_resample:
            pts = resample_points(pts, self.seq_len)

        # 2. Extract movement features
        if self.movement_features is not None:
            if self.movement_features not in ['none', 'cat_move', 'cat_dir', 'all']:
                raise ValueError(
                    f"Unknown movement_features: {self.movement_features}. "
                    f"Must be one of: 'none', 'cat_move', 'cat_dir', 'all'"
                )
            feature_extractor = MovementFeatureExtractor(mode=self.movement_features)
            features = feature_extractor(pts)  # (N, feat_dim)
        else:
            # No feature extraction, use raw xyz directly
            features = pts  # (N, 3)

        # 3. Normalize (using global statistics)
        if self.movement_features == 'all':
            features = (features - GLOBAL_MEAN_12) / GLOBAL_STD_12
        elif self.movement_features == 'cat_dir':
            features = (features - GLOBAL_MEAN_9) / GLOBAL_STD_9
        elif self.movement_features == 'cat_move':
            features = (features - GLOBAL_MEAN_6) / GLOBAL_STD_6
        elif self.movement_features == 'none':
            features = (features - GLOBAL_MEAN_3) / GLOBAL_STD_3
        else: 
            raise ValueError(
                f"Unknown movement_features: {self.movement_features}. "
                f"Must be one of: 'none', 'cat_move', 'cat_dir', 'all'"
            )

        # 4. Padding or truncation to seq_len (if not using resample)
        if self.use_resample:
            # Already resampled to seq_len, all positions are valid
            T = features.shape[0]
            if T != self.seq_len:
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
            # Use padding/truncation
            actual_len = features.shape[0]
            if actual_len > self.seq_len:
                # Truncation
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

        # 5. Convert to batch format
        batch_data = features[np.newaxis, ...]  # (1, seq_len, feat_dim)
        batch_mask = mask[np.newaxis, ...]      # (1, seq_len)

        return batch_data, batch_mask

    def classify(self, testdata) -> int:
        """
        Classify a single 3D digit trajectory.
        Returns the predicted class label (0-9).
        """
        batch_data, batch_mask = self.preprocess(testdata)

        if self.model_type == "supcon":
            output = self.model.forward(batch_data, mask=batch_mask, training=False, return_embeddings=False)
            logits = output
        else:
            logits = self.model.forward(batch_data, mask=batch_mask, training=False)

        predicted_class = int(np.argmax(logits[0]))
        return predicted_class

    def classify_with_confidence(self, testdata):
        """
        Classify a single 3D digit trajectory.
        Returns the predicted class label (0-9), confidence score, and full probability distribution.
        """
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
    digit_classify(testdata)
    ------------------------
    Purpose:
        Classify a single 3D hand-drawn digit trajectory into one of the
        digit classes {0, 1, ..., 9}. This function is the main entry
        point required by the project specification.

    Usage:
        C = digit_classify(testdata)

    Parameters:
        testdata : numpy.ndarray or list-like
            A 2D array of shape (N, 3) representing a single digit
            trajectory, where each row is a 3-D point (x, y, z) sampled
            over time.

    Returns:
        C : int
            The predicted digit label in {0, 1, ..., 9}.
    """
    global _classifier
    if _classifier is None:
        _classifier = DigitClassifier()
    return _classifier.classify(testdata)


def reset_classifier():
    """
    Manually clear the global model cache (e.g., to force reloading weights)
    """
    global _classifier
    _classifier = None