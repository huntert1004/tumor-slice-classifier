import os
import glob
import numpy as np
import nibabel as nib
import cv2
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU(s) available:", gpus)
    except RuntimeError as e:
        print("Could not set memory growth:", e)
else:
    print("No GPU detected, training will use CPU.")

def augment_slice(img, label):
    # img: (H, W, 1)
    # Horizontal flip
    img = tf.image.random_flip_left_right(img)
    # Small brightness/contrast jitter
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img, label

DATA_DIR = "brats24_lite"              # where BraTS24-Lite was downloaded
PREPROC_DIR = "preprocessed_volumes"   # where per-volume .npz files will be stored

TARGET_SIZE = (128, 128)               # slice size
BATCH_SIZE = 16
EPOCHS = 30
TEST_SPLIT = 0.2
RANDOM_STATE = 42


def ensure_dataset():
    if os.path.exists(DATA_DIR):
        print(f"Dataset directory '{DATA_DIR}' already exists, skipping download.")
        return

    print("Downloading BraTS24-Lite from Hugging Face...")
    snapshot_download(
        repo_id="YongchengYAO/BraTS24-Lite",
        repo_type="dataset",
        local_dir=DATA_DIR,
    )
    print("Download complete.")


def resize_slice(slice_2d, target_size):
    """Resize a single 2D slice (H, W) to target_size using bilinear interpolation."""
    return cv2.resize(slice_2d, target_size, interpolation=cv2.INTER_LINEAR)


def preprocess_volumes_to_npz():
    """
    Process each (image, mask) volume pair, slice it, and save per-volume
    slices + labels into PREPROC_DIR/volume_XXXX.npz

    This avoids ever holding all volumes in RAM at once.
    """
    os.makedirs(PREPROC_DIR, exist_ok=True)

    # Find all image and mask files (we know structure now)
    images_pattern = os.path.join(DATA_DIR, "**", "Images-t1c", "*.nii.gz")
    masks_pattern  = os.path.join(DATA_DIR, "**", "Masks",      "*.nii.gz")

    image_files = sorted(glob.glob(images_pattern, recursive=True))
    mask_files  = sorted(glob.glob(masks_pattern,  recursive=True))

    if not image_files or not mask_files:
        raise RuntimeError(
            f"Could not find image/mask volumes. "
            f"Check that structure under '{DATA_DIR}' has Images-t1c/ and Masks/ with .nii.gz files."
        )

    if len(image_files) != len(mask_files):
        print("Warning: number of images and masks differ.")
        print("Images:", len(image_files), "Masks:", len(mask_files))

    print("Found", len(image_files), "volumes")

    for vol_idx, (img_path, msk_path) in enumerate(zip(image_files, mask_files)):
        # Output filename for this volume's slices
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(PREPROC_DIR, f"volume_{vol_idx:04d}_{base_name}.npz")

        # Skip if already preprocessed
        if os.path.exists(out_path):
            print(f"[{vol_idx}] Skipping {base_name} (already preprocessed)")
            continue

        print(f"[{vol_idx}] Processing volume: {base_name}")
        print("   image:", img_path)
        print("   mask :", msk_path)

        # Load the MRI and mask volumes
        img_nii = nib.load(img_path)
        msk_nii = nib.load(msk_path)

        img = img_nii.get_fdata().astype(np.float32)  # (H, W, D)
        msk = msk_nii.get_fdata().astype(np.float32)

        # Per-volume z-score normalization
        img_mean = img.mean()
        img_std = img.std()
        img = (img - img_mean) / (img_std + 1e-8)

        H, W, D = img.shape

        X_slices = []
        y_slices = []

        for z in range(D):
            img_slice = img[:, :, z]
            msk_slice = msk[:, :, z]

            # Label: 1 if any non-zero voxel in mask slice, else 0
            label = 1 if np.any(msk_slice != 0) else 0

            # Optional: downsample non-tumor slices to reduce imbalance
            if label == 0 and np.random.rand() > 0.2:
                continue

            resized = resize_slice(img_slice, TARGET_SIZE)
            resized = np.expand_dims(resized, axis=-1)  # (H, W, 1)

            X_slices.append(resized)
            y_slices.append(label)

        if not X_slices:
            print(f"   WARNING: no slices kept for volume {base_name}, skipping file.")
            continue

        X = np.stack(X_slices, axis=0).astype(np.float32)     # (N_slices, H, W, 1)
        y = np.array(y_slices, dtype=np.int32)                # (N_slices,)

        print(f"   Saved {X.shape[0]} slices "
              f"(tumor: {int(y.sum())}, non-tumor: {int((y == 0).sum())})")

        # Save slices for this volume only
        np.savez_compressed(out_path, X=X, y=y)


def list_preprocessed_volumes():
    """Return sorted list of all preprocessed volume .npz files."""
    pattern = os.path.join(PREPROC_DIR, "volume_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(
            f"No preprocessed volumes found in '{PREPROC_DIR}'. "
            f"Run preprocess_volumes_to_npz() first."
        )
    print(f"Found {len(files)} preprocessed volume files.")
    return files



def npz_to_dataset(path):
    """
    Given a tf.string 'path' to a .npz file, load it and return a Dataset of (X, y) slices.
    """

    def _loader(path_str):
        path_str = path_str.decode("utf-8")
        with np.load(path_str) as data:
            X = data["X"].astype(np.float32)
            y = data["y"].astype(np.int64)
        return X, y

    # Use numpy_function to load npz in a TF graph
    X, y = tf.numpy_function(_loader, [path], [tf.float32, tf.int64])

    # X has shape (N, 128, 128, 1), y has shape (N,)
    X.set_shape([None, TARGET_SIZE[0], TARGET_SIZE[1], 1])
    y.set_shape([None])

    # Turn arrays into a per-slice dataset
    return tf.data.Dataset.from_tensor_slices((X, y))


def make_dataset_from_volume_files(file_paths, training=True):
    file_paths = list(file_paths)
    file_ds = tf.data.Dataset.from_tensor_slices(file_paths)

    if training:
        file_ds = file_ds.shuffle(len(file_paths))

    ds = file_ds.interleave(
        lambda p: npz_to_dataset(p),
        cycle_length=4,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        ds = ds.shuffle(4096)
        ds = ds.map(augment_slice, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model




def main():
    # 1. Make sure raw dataset is present
    ensure_dataset()

    # 2. Preprocess each volume into its own .npz (streaming-friendly)
    preprocess_volumes_to_npz()

    # 3. List all preprocessed volumes
    volume_files = list_preprocessed_volumes()

    # 4. Split volume files into train and validation sets (by volume)
    train_files, val_files = train_test_split(
        volume_files,
        test_size=TEST_SPLIT,
        random_state=RANDOM_STATE,
    )

    print("Train volumes:", len(train_files), "Val volumes:", len(val_files))

    # 5. Build streaming datasets
    train_ds = make_dataset_from_volume_files(train_files, training=True)
    val_ds   = make_dataset_from_volume_files(val_files,   training=False)

    # 6. Build and train the model
    model = build_model(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 1))
    model.summary()
    callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # 7. Save the model
    os.makedirs("models", exist_ok=True)
    model.save("models/tumor_slice_classifier_streaming.h5")
    print("Model saved to models/tumor_slice_classifier_streaming.h5")


if __name__ == "__main__":
    main()
