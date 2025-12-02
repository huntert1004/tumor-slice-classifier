import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# Import helpers + config from training script
from train import (
    list_preprocessed_volumes,
    make_dataset_from_volume_files,
    TEST_SPLIT,
    RANDOM_STATE,
)

MODEL_PATH = "models/tumor_slice_classifier_streamingv1.1.h5"



def load_model():
    """Load the trained streaming model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            f"Train the model first by running train_streaming.py."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
    return model


def build_val_dataset():
    """
    Rebuild the validation dataset from preprocessed .npz volume files.
    We split the *files* the same way as in training.
    """
    volume_files = list_preprocessed_volumes()

    # Same random_state + test_size as in training
    _, val_files = train_test_split_volume_files(volume_files)

    print("Validation volumes:", len(val_files))
    val_ds = make_dataset_from_volume_files(val_files, training=False)
    return val_ds


def train_test_split_volume_files(volume_files):
    """Helper to split volume file list using sklearn's train_test_split."""
    from sklearn.model_selection import train_test_split

    train_files, val_files = train_test_split(
        volume_files,
        test_size=TEST_SPLIT,
        random_state=RANDOM_STATE,
    )
    return train_files, val_files


def collect_predictions(model, val_ds):
    """Run model over entire val_ds, return y_true, y_prob, y_pred."""
    all_y = []
    all_p = []

    for batch_imgs, batch_labels in val_ds:
        y = batch_labels.numpy()
        p = model.predict(batch_imgs, verbose=0).flatten()
        all_y.append(y)
        all_p.append(p)

    y_true = np.concatenate(all_y, axis=0)
    y_prob = np.concatenate(all_p, axis=0)
    y_pred = (y_prob >= 0.5).astype(np.int32)  # threshold 0.5

    return y_true, y_prob, y_pred


def print_full_metrics(y_true, y_pred, y_prob):
    print("\nClassification report (threshold = 0.5):")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    auc = roc_auc_score(y_true, y_prob)
    print(f"\nROC AUC (using probabilities): {auc:.4f}")


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve (tumor vs non-tumor slices)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def show_examples_by_type(model, val_ds, num_per_type=6):
    """
    Show examples of:
      - true positives (TP)
      - false positives (FP)
      - true negatives (TN)
      - false negatives (FN)
    """
    tps, fps, tns, fns = [], [], [], []

    for batch_imgs, batch_labels in val_ds:
        imgs = batch_imgs.numpy()
        labels = batch_labels.numpy()
        probs = model.predict(imgs, verbose=0).flatten()
        preds = (probs >= 0.5).astype(np.int32)

        for img, y, p, yhat in zip(imgs, labels, probs, preds):
            if y == 1 and yhat == 1 and len(tps) < num_per_type:
                tps.append((img, y, p))
            elif y == 0 and yhat == 1 and len(fps) < num_per_type:
                fps.append((img, y, p))
            elif y == 0 and yhat == 0 and len(tns) < num_per_type:
                tns.append((img, y, p))
            elif y == 1 and yhat == 0 and len(fns) < num_per_type:
                fns.append((img, y, p))

        if (len(tps) >= num_per_type and len(fps) >= num_per_type and
            len(tns) >= num_per_type and len(fns) >= num_per_type):
            break

    def _plot_group(examples, title):
        if not examples:
            print(f"No examples for {title}")
            return
        imgs, ys, ps = zip(*examples)
        imgs = np.array(imgs)
        ps = np.array(ps)

        cols = 3
        rows = int(np.ceil(len(imgs) / cols))
        plt.figure(figsize=(4 * cols, 4 * rows))
        for i, (img, prob) in enumerate(zip(imgs, ps)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img[:, :, 0], cmap="gray")
            plt.axis("off")
            plt.title(f"P: {prob:.2f}")
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    _plot_group(tps, "True Positives (tumor, predicted tumor)")
    _plot_group(fps, "False Positives (no tumor, predicted tumor)")
    _plot_group(tns, "True Negatives (no tumor, predicted no tumor)")
    _plot_group(fns, "False Negatives (tumor, predicted no tumor)")



def main():
    model = load_model()

    # Build validation dataset and collect predictions
    val_ds = build_val_dataset()
    y_true, y_prob, y_pred = collect_predictions(model, val_ds)

    # Print metrics + confusion matrix
    print_full_metrics(y_true, y_pred, y_prob)

    # Plot ROC curve
    plot_roc_curve(y_true, y_prob)

    # Rebuild val_ds for visualization (we already iterated it above)
    val_ds = build_val_dataset()

    # Show example slices by type
    show_examples_by_type(model, val_ds, num_per_type=6)


if __name__ == "__main__":
    main()
