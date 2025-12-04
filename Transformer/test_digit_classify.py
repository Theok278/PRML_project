import numpy as np
from pathlib import Path
from digit_classify import digit_classify, reset_classifier
import time


def find_stroke_files():
    """
    find all stroke_*_*.csv files in the current directory,
    and parse out the digit labels from filenames (format: stroke_<label>_<id>.csv)
    return a list: [(path, label), ...]
    """
    stroke_files = []

    for path in sorted(Path("../../digits_3d/training_data").glob("stroke_*_*.csv")):
        name = path.name  # e.g. "stroke_0_0001.csv"
        parts = name.split("_")
        if len(parts) < 3:
            # Not the expected format, skip
            continue

        # parts[0] = "stroke"
        # parts[1] = "<label>"
        # parts[2] = "<id>.csv"
        try:
            label = int(parts[1])
        except ValueError:
            # Failed to parse label, skip
            continue

        stroke_files.append((path, label))

    return stroke_files


def evaluate_digit_classifier(num_classes: int = 10, verbose: bool = True):
    """
    Evaluate all stroke_*_*.csv files in the current directory,
    computing accuracy and confusion matrix.
    """
    start_time = time.time()
    # Ensure a clean state at the start of each run (optional)
    reset_classifier()

    samples = find_stroke_files()
    if not samples:
        print("No stroke_*_*.csv files found in current directory.")
        return

    print(f"Found {len(samples)} stroke files.")

    # Confusion matrix: rows = true labels, columns = predicted labels
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    correct = 0
    total = 0

    for idx, (path, true_label) in enumerate(samples, start=1):
        # Call your digit_classify (it will automatically load checkpoint and cache model)
        pred = digit_classify(str(path))

        # Update confusion matrix (prevent out-of-range)
        if 0 <= true_label < num_classes and 0 <= pred < num_classes:
            confusion[true_label, pred] += 1
        else:
            print(
                f"Warning: label or prediction out of range for file {path.name}: "
                f"true_label={true_label}, pred={pred}"
            )

        if pred == true_label:
            correct += 1
        total += 1

        if verbose and idx % 50 == 0:
            print(f"Processed {idx} samples...")
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nEvaluation completed in {elapsed:.2f} seconds.")

    accuracy = correct / total if total > 0 else 0.0

    print("\n=== Evaluation Result ===")
    print(f"Total samples : {total}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy:.4f}")

    # Print confusion matrix
    print("\nConfusion matrix (rows = true labels, cols = predicted labels):\n")

    # Print header
    header = "      " + " ".join(f"{c:4d}" for c in range(num_classes))
    print(header)
    print("     " + "-" * (5 * num_classes))

    # 每一行：真实标签 -> 各预测计数
    for i in range(num_classes):
        row_counts = " ".join(f"{confusion[i, j]:4d}" for j in range(num_classes))
        print(f"{i:2d} | {row_counts}")

    print("\nDone.")


if __name__ == "__main__":
    evaluate_digit_classifier()