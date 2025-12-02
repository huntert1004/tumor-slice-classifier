import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Import helpers + config from training script
from train import (
    list_preprocessed_volumes,
    make_dataset_from_volume_files,
    PREPROC_DIR,
    TEST_SPLIT,
    RANDOM_STATE,
    TARGET_SIZE,
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
    We split the *files* the same way as in training, then create a
    streaming tf.data.Dataset for validation only.
    """
    volume_files = list_preprocessed_volumes()

    _, val_files = train_test_split(
        volume_files,
        test_size=TEST_SPLIT,
        random_state=RANDOM_STATE,
    )

    print("Validation volumes:", len(val_files))

    val_ds = make_dataset_from_volume_files(val_files, training=False)
    return val_ds


def evaluate_model(model, val_ds):
    """Evaluate the model on the validation dataset."""
    results = model.evaluate(val_ds, verbose=1)
    print("\nValidation results:")
    for name, value in zip(model.metrics_names, results):
        print(f"  {name}: {value:.4f}")


def show_tumor_examples(model, val_ds, num_samples=8):
    """
    Scan through val_ds and collect slices where the true label == 1 (tumor),
    then show up to num_samples of them with predicted probabilities.
    """
    tumor_imgs = []
    tumor_labels = []

    # Iterate over batches from the validation dataset
    for batch_imgs, batch_labels in val_ds:
        batch_imgs_np = batch_imgs.numpy()
        batch_labels_np = batch_labels.numpy()

        # Mask for tumor slices
        mask = batch_labels_np == 1
        if np.any(mask):
            tumor_imgs.append(batch_imgs_np[mask])
            tumor_labels.append(batch_labels_np[mask])

        # Stop once we have enough
        total_collected = sum(x.shape[0] for x in tumor_imgs)
        if total_collected >= num_samples:
            break

    if not tumor_imgs:
        print("No tumor slices found in validation set.")
        return

    # Stack and trim to exactly num_samples
    imgs = np.concatenate(tumor_imgs, axis=0)[:num_samples]
    labels = np.concatenate(tumor_labels, axis=0)[:num_samples]

    # Get predictions
    preds = model.predict(imgs)
    preds = preds.flatten()

    cols = 4
    rows = int(np.ceil(num_samples / cols))

    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, (img, true_label, pred_prob) in enumerate(zip(imgs, labels, preds)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img[:, :, 0], cmap="gray")
        plt.axis("off")
        plt.title(f"T: {int(true_label)}, P: {pred_prob:.2f}")
    plt.tight_layout()
    plt.show()


def main():
    model = load_model()
    val_ds = build_val_dataset()

    evaluate_model(model, val_ds)

    # Rebuild val_ds because we just iterated it inside evaluate_model()
    val_ds = build_val_dataset()

    show_tumor_examples(model, val_ds, num_samples=8)


if __name__ == "__main__":
    main()
