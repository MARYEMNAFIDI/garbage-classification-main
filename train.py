import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (32, 32)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def build_model() -> tf.keras.Model:
    """Define the CNN used for garbage classification."""
    model = models.Sequential(
        [
            layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def make_datagens(data_dir: Path, batch_size: int):
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        subset="training",
        seed=42,
    )
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        subset="validation",
        seed=42,
    )

    # Ensure classes align with expectations to avoid silent label swaps.
    classes = train_gen.class_indices
    if list(classes.keys()) != CLASS_NAMES:
        raise ValueError(f"Class order mismatch. Found {list(classes.keys())}, expected {CLASS_NAMES}")

    return train_gen, val_gen


def load_or_build(resume_from: Optional[Path]) -> tf.keras.Model:
    if resume_from is not None and resume_from.exists():
        model = tf.keras.models.load_model(resume_from)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model
    return build_model()


def train(data_dir: Path, batch_size: int, epochs: int, output: Path, resume_from: Optional[Path]):
    train_gen, val_gen = make_datagens(data_dir, batch_size)
    model = load_or_build(resume_from)
    output_path = str(output)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(output_path, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Always save the final model (best checkpoint already saved by ModelCheckpoint).
    model.save(output_path)

    # Persist training history for quick inspection.
    history_out = output.with_suffix(".history.json")
    history_out.write_text(json.dumps(history.history, indent=2))
    print(f"Training complete. Model saved to {output} and history to {history_out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train or fine-tune the garbage classifier CNN.")
    parser.add_argument("--data-dir", type=Path, default=Path("Garbage/original_images"), help="Directory with class subfolders.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_retrained.h5"),
        help="Where to write the trained model (H5 format).",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional existing model path to continue training from (e.g., model.h5).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)
    train(args.data_dir, args.batch_size, args.epochs, args.output, args.resume_from)


if __name__ == "__main__":
    main()
