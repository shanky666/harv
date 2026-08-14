"""
Unified Training Script for HarvestLenz Per-Fruit Grading Models

Usage:
    python train.py --fruit mango
    python train.py --fruit pineapple
    python train.py --fruit grapes
    python train.py --fruit pomegranate
    python train.py --fruit potato --epochs 30 --batch-size 16

Architecture (Fruits):
    MobileNetV2 (ImageNet pretrained) + GlobalAveragePooling2D + Dense(256) + BatchNorm
    + Dropout(0.4) + Dense(128) + Dropout(0.3) + Dense(num_classes, softmax)

Architecture (Vegetables — potato, carrot, etc.):
    EfficientNetB0 (ImageNet pretrained) + GlobalAveragePooling2D + Dense(256) + BatchNorm
    + Dropout(0.4) + Dense(128) + Dropout(0.3) + Dense(num_classes, softmax)

Output:
    app/models/weights/{fruit}.keras
"""
import os
import sys
import argparse
import shutil
import random
import json
from pathlib import Path

import numpy as np

ALL_CLASSES = ["Better", "Good", "Reject"]
POTATO_DISEASE_CLASSES = [
    "Black Scurf", "Blackleg", "Blackspot Bruising", "Brown Rot",
    "Common Scab", "Dry Rot", "Healthy Potatoes", "Miscellaneous",
    "Pink Rot", "Soft Rot",
]
FRUIT_DATASETS = {
    "mango": "dataset_mango/dataset_mango",
    "pineapple": "dataset-pineapple/dataset-pineapple",
    "grapes": "dataset-grapes/dataset-grapes",
    "pomegranate": "dataset-pomegranate/dataset-pomegranate",
    "potato": "dataset-potato/dataset-potato",
    "orange": "dataset-orange/dataset-orange",
    "guava": "dataset-guava/dataset-guava",
    "kiwi": "dataset-kiwi-v2/dataset-kiwi",
    "capsicum": "dataset-capsicum/dataset-capsicum",
    "cucumber": "dl/agrifresh_processed.zip",
    "tomato": "dl/agrifresh_processed.zip",
    "carrot": "dl/fvd_healthy_vs_rotten.zip",
    "watermelon": "dl/freshness50.zip",
    "banana": "dl/fruits_fresh_rotten.zip",
    "cocoa": "dl/cocoa",
    "coffee": "grading_split_coffee",
    "strawberry": "dataset-strawberry/dataset-strawberry",
    "plum": "grading_split_plum",
    "peach": "grading_split_peach",
    "pear": "grading_split_pear",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a per-fruit grading model")
    parser.add_argument("--fruit", type=str, required=True,
                        choices=list(FRUIT_DATASETS.keys()),
                        help="Fruit type to train on")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Total epochs (phase 1: 10 frozen, phase 2: remaining fine-tuned)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for weights (default: app/models/weights/)")
    parser.add_argument("--split-dir", type=str, default=None,
                        help="Pre-split dataset directory (train/val/test)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    return parser.parse_args()


def create_data_splits(dataset_path: str, split_dir: str, classes: list[str] | None = None, train_ratio=0.8, val_ratio=0.1):
    """
    Creates train/val/test splits from a folder-per-class dataset.
    Skips if splits already exist.
    """
    train_dir = os.path.join(split_dir, "train")
    val_dir = os.path.join(split_dir, "val")
    test_dir = os.path.join(split_dir, "test")

    if os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir):
        train_count = sum(len(files) for _, _, files in os.walk(train_dir))
        val_count = sum(len(files) for _, _, files in os.walk(val_dir))
        if train_count > 0 and val_count > 0:
            print(f"Splits already exist: train={train_count}, val={val_count}")
            return train_dir, val_dir, test_dir

    if classes is None:
        classes = sorted([
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ])
        print(f"Auto-detected {len(classes)} class directories: {classes}")

    print(f"Creating data splits from {dataset_path}...")
    for cls in classes:
        cls_dir = os.path.join(dataset_path, cls)
        if not os.path.exists(cls_dir):
            print(f"  WARNING: Class directory not found: {cls_dir}")
            continue

        images = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            split_cls_dir = os.path.join(split_dir, split_name, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            for img_name in split_images:
                src = os.path.join(cls_dir, img_name)
                dst = os.path.join(split_cls_dir, img_name)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

        print(f"  {cls}: {n} images -> train={n_train}, val={n_val}, test={n - n_train - n_val}")

    return train_dir, val_dir, test_dir


_backend_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _backend_root)
from app.models.shared_mobilenet import build_grading_model, build_vegetable_model, FRUIT_NUM_CLASSES, VEGETABLE_BACKBONES


def _make_preprocessor(fruit):
    from app.models.shared_mobilenet import VEGETABLE_BACKBONES
    backbone = VEGETABLE_BACKBONES.get(fruit, "mobilenet")
    if backbone == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input as fn
    else:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as fn

    def preprocess(x):
        return fn(x.astype("float32"))

    return preprocess


def get_augmented_generator(img_size, batch_size, fruit=None):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_aug = ImageDataGenerator(
        preprocessing_function=_make_preprocessor(fruit),
        rotation_range=25,
        brightness_range=(0.7, 1.3),
        zoom_range=[0.85, 1.15],
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=6,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest",
    )

    val_aug = ImageDataGenerator(
        preprocessing_function=_make_preprocessor(fruit),
    )

    return train_aug, val_aug


def get_plain_generator(img_size, batch_size, fruit=None):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    aug = ImageDataGenerator(
        preprocessing_function=_make_preprocessor(fruit),
    )
    return aug, aug


def train(args):
    print(f"\n{'='*60}")
    print(f"  HarvestLenz Model Trainer")
    print(f"  Fruit: {args.fruit.upper()}")
    print(f"  Epochs: {args.epochs} (Phase 1: 10 + Phase 2: {max(0, args.epochs - 10)})")
    print(f"  Batch Size: {args.batch_size}")
    print(f"{'='*60}\n")

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    if args.split_dir:
        train_dir = os.path.join(args.split_dir, "train")
        val_dir = os.path.join(args.split_dir, "val")
    else:
        dataset_rel = FRUIT_DATASETS[args.fruit]
        dataset_path = os.path.join(workspace_root, dataset_rel)
        split_dir = os.path.join(workspace_root, f"grading_split_{args.fruit}")
        class_names = POTATO_DISEASE_CLASSES if args.fruit == "potato" else None
        train_dir, val_dir, _ = create_data_splits(dataset_path, split_dir, classes=class_names)

    if not os.path.exists(train_dir):
        print(f"ERROR: Training directory not found: {train_dir}")
        sys.exit(1)

    if args.split_dir:
        candidate_classes = sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ])
        print(f"Auto-detected classes from split dir: {candidate_classes}")
    else:
        candidate_classes = POTATO_DISEASE_CLASSES if args.fruit == "potato" else ALL_CLASSES
    actual_classes = []
    for cls in candidate_classes:
        cls_dir = os.path.join(train_dir, cls)
        n = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]) if os.path.isdir(cls_dir) else 0
        if n > 0:
            actual_classes.append(cls)

    num_classes = len(actual_classes)
    print(f"Detected {num_classes} non-empty classes: {actual_classes}")

    img_size = args.input_size
    batch_size = args.batch_size

    if args.no_augment:
        train_gen, val_gen = get_plain_generator(img_size, batch_size, args.fruit)
    else:
        train_gen, val_gen = get_augmented_generator(img_size, batch_size, args.fruit)

    train_flow = train_gen.flow_from_directory(
        train_dir, target_size=(img_size, img_size), batch_size=batch_size,
        class_mode="categorical", classes=actual_classes, shuffle=True,
    )
    val_flow = val_gen.flow_from_directory(
        val_dir, target_size=(img_size, img_size), batch_size=batch_size,
        class_mode="categorical", classes=actual_classes, shuffle=False,
    )

    print(f"\nClass indices: {train_flow.class_indices}")
    print(f"Training samples: {train_flow.samples}")
    print(f"Validation samples: {val_flow.samples}")

    if VEGETABLE_BACKBONES.get(args.fruit) == "efficientnet":
        model = build_vegetable_model(num_classes=num_classes)
        print(f"  Using EfficientNetB0 backbone for {args.fruit}")
        base_layer_name = "efficientnetb0"
    else:
        model = build_grading_model(num_classes=num_classes)
        print(f"  Using MobileNetV2 backbone for {args.fruit}")
        base_layer_name = "mobilenetv2_1.00_224"

    # Locate the backbone sub-model inside the Functional graph
    base_model = None
    for layer in model.layers:
        if layer.name.startswith("mobilenetv2") or layer.name.startswith("efficientnetb0"):
            base_model = layer
            break
    if base_model is None:
        # Fallback: treat the whole model as base (won't unfreeze selectively)
        base_model = model

    output_dir = args.output_dir or os.path.join(workspace_root, "backend", "backend", "app", "models", "weights")
    os.makedirs(output_dir, exist_ok=True)

    phase1_epochs = min(10, args.epochs)
    phase2_epochs = max(0, args.epochs - phase1_epochs)

    # Compile Phase 1 (base frozen, head only)
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n--- Phase 1: Head training ({phase1_epochs} epochs, base frozen) ---")
    history1 = model.fit(
        train_flow,
        epochs=phase1_epochs,
        validation_data=val_flow,
        verbose=1,
    )

    if phase2_epochs > 0:
        print(f"\n--- Phase 2: Fine-tuning ({phase2_epochs} epochs, unfreezing from layer 100) ---")
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr * 0.1),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        history2 = model.fit(
            train_flow,
            epochs=phase1_epochs + phase2_epochs,
            initial_epoch=phase1_epochs,
            validation_data=val_flow,
            verbose=1,
        )

    output_path = os.path.join(output_dir, f"{args.fruit}.keras")
    model.save(output_path)
    print(f"\nModel saved to: {output_path}")

    h5_path = os.path.join(output_dir, f"{args.fruit}.h5")
    try:
        model.save(h5_path)
        print(f"H5 backup saved to: {h5_path}")
    except Exception as e:
        print(f"H5 save failed (non-critical): {e}")

    val_loss, val_acc = model.evaluate(val_flow, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc:.4f}")
    print(f"Final validation loss: {val_loss:.4f}")

    history_path = os.path.join(output_dir, f"{args.fruit}_history.json")
    all_acc = list(history1.history.get("accuracy", []))
    all_val_acc = list(history1.history.get("val_accuracy", []))
    all_loss = list(history1.history.get("loss", []))
    all_val_loss = list(history1.history.get("val_loss", []))
    if phase2_epochs > 0:
        all_acc.extend(history2.history.get("accuracy", []))
        all_val_acc.extend(history2.history.get("val_accuracy", []))
        all_loss.extend(history2.history.get("loss", []))
        all_val_loss.extend(history2.history.get("val_loss", []))
    with open(history_path, "w") as f:
        json.dump({
            "accuracy": [float(x) for x in all_acc],
            "val_accuracy": [float(x) for x in all_val_acc],
            "loss": [float(x) for x in all_loss],
            "val_loss": [float(x) for x in all_val_loss],
        }, f, indent=2)

    print(f"\nTraining complete. Weight file: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
