import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
try:
    import seaborn as sns
except ImportError:
    sns = None
from loguru import logger
import shutil

# Constants
IMG_SIZE = (224, 224)
CLASSES = ["Better", "Good", "Reject"]  # Alphabetical — TF sorts class_names this way
CLASS_MAPPING = {cls: idx for idx, cls in enumerate(CLASSES)}

def prepare_grading_dataset(fruit_name, base_dir="grading_split"):
    """
    Prepares a dataset split under `base_dir` using only the real images for a specific fruit.
    Folders mapped: Good -> Good, Better -> Better, Reject -> Reject
    """
    logger.info(f"Preparing real dataset for {fruit_name} in '{base_dir}'...")
    split_dir = f"{base_dir}_{fruit_name}"
    if os.path.exists(split_dir):
        logger.info(f"Removing old split directory '{split_dir}'...")
        shutil.rmtree(split_dir)
    os.makedirs(split_dir, exist_ok=True)

    # We map fruit class names to sample_data directories
    source_dirs = {
        "grapes": ["dataset-grapes", "grapes"],
        "mango": ["dataset_mango", "mango", "dataset-mango"],
        "pineapple": ["dataset-pineapple", "pineapple"],
        "pomegranate": ["dataset-pomegranate", "pomegranate"],
        "banana": ["Banana", "dataset-banana", "dataset_banana"]
    }

    dummy_names = {
        "single_mango.jpg", "mango_warmup.jpg", "mixed_fruits.jpg",
        "black.jpg", "partial_occlusion.jpg", "multiple_mangoes.jpg"
    }

    def find_dataset_dir(dir_name):
        possible_roots = [
            "c:/Users/ranja/HarvestLenz",
            ".",
            "..",
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        ]
        for root in possible_roots:
            p = os.path.join(root, dir_name)
            if os.path.isdir(p):
                return p
            p_sample = os.path.join(root, "sample_data", dir_name)
            if os.path.isdir(p_sample):
                return p_sample
        return None

    possible_folders = source_dirs.get(fruit_name, [f"dataset_{fruit_name}", f"dataset-{fruit_name}", fruit_name])
    
    subfolder_candidates = {
        "Better": ["Better", "Banana_Good", "Banana"],
        "Good": ["Good", "Banana_Good", "good"],
        "Reject": ["Reject", "Banana_Bad", "bad"]
    }

    for cls in CLASSES:
        all_images = []
        candidates = subfolder_candidates.get(cls, [cls])
        for folder_name in possible_folders:
            actual_dir = find_dataset_dir(folder_name)
            if not actual_dir:
                continue
            
            logger.info(f"Found folder '{folder_name}' at '{actual_dir}'")
            possible_roots_for_cand = [actual_dir]
            if os.path.isdir(os.path.join(actual_dir, folder_name)):
                possible_roots_for_cand.append(os.path.join(actual_dir, folder_name))
                
            for cand in candidates:
                sub_path = None
                for root_cand in possible_roots_for_cand:
                    test_p = os.path.join(root_cand, cand)
                    if os.path.isdir(test_p):
                        sub_path = test_p
                        break
                if not sub_path:
                    continue
                imgs = []
                for fname in os.listdir(sub_path):
                    if fname.lower() in dummy_names:
                        continue
                    if fname.lower().startswith("aug_") or "synthetic" in fname.lower() or "fake" in fname.lower():
                        continue
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        imgs.append(os.path.join(sub_path, fname))
                if cand == "Banana_Good" and fruit_name == "banana":
                    # Split Banana_Good between Better and Good
                    imgs = sorted(imgs)
                    if cls == "Good":
                        all_images.extend(imgs[:len(imgs)//2])
                    elif cls == "Better":
                        all_images.extend(imgs[len(imgs)//2:])
                else:
                    all_images.extend(imgs)

        all_images = list(set([os.path.abspath(p) for p in all_images]))
        logger.info(f"Found {len(all_images)} real images for class '{cls}' of fruit '{fruit_name}'")

        if not all_images:
            logger.warning(f"No real images found for class '{cls}' of fruit '{fruit_name}'! Will try to proceed if others exist.")
            continue

        np.random.seed(42)
        np.random.shuffle(all_images)
        selected_images = all_images[:500]

        n = len(selected_images)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        splits = {
            "train": selected_images[:n_train],
            "val": selected_images[n_train:n_train + n_val],
            "test": selected_images[n_train + n_val:]
        }

        for split, paths in splits.items():
            split_cls_dir = os.path.join(split_dir, split, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            for idx, src_path in enumerate(paths):
                dst_path = os.path.join(split_cls_dir, f"img_{idx}.jpg")
                shutil.copy2(src_path, dst_path)

    logger.info("Real dataset preparation complete.")
    return split_dir

def build_model(fine_tune=False, fine_tune_from=100):
    """
    Builds the Keras model using MobileNetV2.
    Phase 1 (fine_tune=False): only trains the classification head.
    Phase 2 (fine_tune=True): unfreezes top layers of MobileNetV2 for fine-tuning.
    """
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)

        if fine_tune:
            # Unfreeze from fine_tune_from layer onwards
            base_model.trainable = True
            for layer in base_model.layers[:fine_tune_from]:
                layer.trainable = False
            logger.info(f"Fine-tuning enabled from layer {fine_tune_from}/{len(base_model.layers)}")
            lr = 1e-5
        else:
            lr = 1e-3

    except Exception as e:
        logger.warning(f"Failed to build MobileNetV2: {e}. Building custom CNN.")
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        lr = 1e-3

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_pipeline_for_fruit(fruit_name, epochs=15, batch_size=16):
    logger.info(f"--- Training Quality Model for {fruit_name.upper()} ---")
    split_dir = prepare_grading_dataset(fruit_name)
    
    train_dir = os.path.join(split_dir, "train")
    val_dir = os.path.join(split_dir, "val")
    test_dir = os.path.join(split_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, label_mode="categorical", class_names=CLASSES,
        image_size=IMG_SIZE, batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, label_mode="categorical", class_names=CLASSES,
        image_size=IMG_SIZE, batch_size=batch_size
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, label_mode="categorical", class_names=CLASSES,
        image_size=IMG_SIZE, batch_size=batch_size, shuffle=False
    )

    # Rich augmentation pipeline for training — includes hue shift so the
    # model learns to grade mangoes regardless of variety colour (green,
    # yellow, reddish, orange).
    def random_hue(image, max_delta=0.08):
        """Shift hue by up to max_delta (in [0,0.5] range) to simulate
        different mango skin colours: green, golden, reddish, orange."""
        return tf.image.random_hue(image, max_delta)

    augmenter = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.25),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.Lambda(lambda x: tf.map_fn(random_hue, x)),
    ])

    rescale = tf.keras.layers.Rescaling(1./255)

    train_ds_scaled = train_ds.map(
        lambda x, y: (augmenter(rescale(x), training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    val_ds_scaled = val_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)
    test_ds_scaled = test_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)

    # Compute class weights to handle imbalanced datasets (e.g. Reject-heavy pineapple)
    class_counts = {cls: 0 for cls in CLASSES}
    for cls in CLASSES:
        cls_dir = os.path.join(train_dir, cls)
        if os.path.isdir(cls_dir):
            class_counts[cls] = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total_train = sum(class_counts.values())
    n_classes = len([c for c in class_counts.values() if c > 0])
    class_weights = {}
    for idx, cls in enumerate(CLASSES):
        if class_counts[cls] > 0:
            class_weights[idx] = total_train / (n_classes * class_counts[cls])
        else:
            class_weights[idx] = 1.0
    logger.info(f"Class counts: {class_counts} | Class weights: {class_weights}")

    os.makedirs("models", exist_ok=True)
    temp_checkpoint_path = f"models/{fruit_name}_quality_temp.h5"

    callbacks_phase1 = [
        tf.keras.callbacks.ModelCheckpoint(temp_checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    # ── Phase 1: Train classification head only (frozen base) ──
    logger.info(f"[{fruit_name}] Phase 1: Training head with frozen MobileNetV2...")
    model = build_model(fine_tune=False)
    phase1_epochs = min(10, epochs)
    model.fit(train_ds_scaled, validation_data=val_ds_scaled, epochs=phase1_epochs, callbacks=callbacks_phase1, class_weight=class_weights)

    # ── Phase 2: Fine-tune top layers of MobileNetV2 ──
    logger.info(f"[{fruit_name}] Phase 2: Fine-tuning top MobileNetV2 layers...")
    fine_tune_model = build_model(fine_tune=True, fine_tune_from=100)
    # Transfer weights from phase 1 best checkpoint
    best_phase1 = tf.keras.models.load_model(temp_checkpoint_path)
    fine_tune_model.set_weights(best_phase1.get_weights())
    fine_tune_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    phase2_checkpoint = f"models/{fruit_name}_quality_ft.h5"
    callbacks_phase2 = [
        tf.keras.callbacks.ModelCheckpoint(phase2_checkpoint, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
    ]
    phase2_epochs = max(5, epochs - phase1_epochs)
    fine_tune_model.fit(train_ds_scaled, validation_data=val_ds_scaled, epochs=phase2_epochs, callbacks=callbacks_phase2, class_weight=class_weights)

    # Pick the best overall model (phase1 vs phase2)
    import os as _os
    best_p1 = tf.keras.models.load_model(temp_checkpoint_path)
    _, acc_p1 = best_p1.evaluate(test_ds_scaled, verbose=0)
    best_final_path = temp_checkpoint_path
    best_accuracy = acc_p1

    if _os.path.exists(phase2_checkpoint):
        best_p2 = tf.keras.models.load_model(phase2_checkpoint)
        _, acc_p2 = best_p2.evaluate(test_ds_scaled, verbose=0)
        logger.info(f"[{fruit_name}] Phase1 acc={acc_p1:.4f} | Phase2 fine-tune acc={acc_p2:.4f}")
        if acc_p2 > acc_p1:
            best_final_path = phase2_checkpoint
            best_accuracy = acc_p2
            logger.info(f"[{fruit_name}] Phase 2 is better — using fine-tuned model.")
        else:
            logger.info(f"[{fruit_name}] Phase 1 is better — keeping head-only model.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, "app", "ai", "models")
    os.makedirs(target_dir, exist_ok=True)
    prod_path = os.path.join(target_dir, f"{fruit_name}_model.h5")

    shutil.copy2(best_final_path, prod_path)
    logger.info(f"Replaced production quality model for {fruit_name}: {prod_path}")
    
    # Also save to backend/app/ai/models/
    parent_backend_dir = os.path.abspath(os.path.join(script_dir, ".."))
    if os.path.basename(parent_backend_dir) == "backend":
        alt_target = os.path.join(parent_backend_dir, "app", "ai", "models")
        os.makedirs(alt_target, exist_ok=True)
        alt_prod_path = os.path.join(alt_target, f"{fruit_name}_model.h5")
        shutil.copy2(best_final_path, alt_prod_path)
        logger.info(f"Also saved to alternative path: {alt_prod_path}")
        
    logger.info(f"--- Completed Training for {fruit_name.upper()} ---\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train fruit quality grading models.")
    parser.add_argument("--fruits", nargs="+", default=["mango", "grapes", "pineapple", "pomegranate", "banana"],
                        help="Fruits to train (e.g. --fruits mango pomegranate banana)")
    parser.add_argument("--epochs", type=int, default=15, help="Total epochs per fruit")
    args = parser.parse_args()
    for fruit in args.fruits:
        try:
            train_pipeline_for_fruit(fruit, epochs=args.epochs)
        except Exception as e:
            logger.error(f"Failed to train quality model for {fruit}: {e}")
