"""
Quick retrain: load existing mango model, fine-tune more epochs on original split.
"""
import os, numpy as np, tensorflow as tf, shutil, json
from loguru import logger
from sklearn.metrics import classification_report
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

FRUIT = "mango"
CLASSES = ["Better", "Good", "Reject"]
IMG_SIZE = (224, 224)
SPLIT_DIR = "grading_split_mango"

# Check existing split
if not os.path.exists(os.path.join(SPLIT_DIR, "train")):
    print("ERROR: grading_split_mango not found. Run train_grading_model.py first.")
    exit(1)

# Print class counts
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        d = os.path.join(SPLIT_DIR, split, cls)
        n = len(os.listdir(d)) if os.path.exists(d) else 0
        print(f"  {split}/{cls}: {n}")

BATCH = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, "train"), label_mode="categorical",
    class_names=CLASSES, image_size=IMG_SIZE, batch_size=BATCH)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, "val"), label_mode="categorical",
    class_names=CLASSES, image_size=IMG_SIZE, batch_size=BATCH)
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, "test"), label_mode="categorical",
    class_names=CLASSES, image_size=IMG_SIZE, batch_size=BATCH, shuffle=False)

augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
])
rescale = tf.keras.layers.Rescaling(1./255)

train_scaled = train_ds.map(lambda x,y: (augmenter(rescale(x), training=True), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_scaled = val_ds.map(lambda x,y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)
test_scaled = test_ds.map(lambda x,y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)

# Class weights
counts = {}
for cls in CLASSES:
    d = os.path.join(SPLIT_DIR, "train", cls)
    counts[cls] = len([f for f in os.listdir(d) if f.endswith(('.jpg','.jpeg','.png'))])
total = sum(counts.values())
n_cls = len([c for c in counts.values() if c > 0])
cw = {i: total/(n_cls*counts[c]) for i,c in enumerate(CLASSES)}
logger.info(f"Counts: {counts}, Weights: {cw}")

# Load existing best model and continue fine-tuning
BEST_PATH = f"models/{FRUIT}_quality_ft.h5"
TEMP_PATH = f"models/{FRUIT}_quality_temp.h5"

if os.path.exists(BEST_PATH):
    logger.info(f"Loading existing model from {BEST_PATH}")
    model = tf.keras.models.load_model(BEST_PATH)
else:
    logger.info("No existing model, building fresh")
    base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
    base.trainable = True
    for l in base.layers[:100]: l.trainable = False
    inp = tf.keras.Input((224,224,3))
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inp, out)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(TEMP_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
]

logger.info("Fine-tuning for 20 more epochs...")
model.fit(train_scaled, validation_data=val_scaled, epochs=20, callbacks=callbacks, class_weight=cw)

# Evaluate
best = tf.keras.models.load_model(TEMP_PATH)
_, acc = best.evaluate(test_scaled, verbose=0)
logger.info(f"Test accuracy: {acc:.4f}")

y_true, y_pred = [], []
for bx, by in test_scaled:
    preds = best.predict(bx, verbose=0)
    y_true.extend(np.argmax(by.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

report = classification_report(y_true, y_pred, target_names=CLASSES)
logger.info(f"\n{report}")

cm = __import__('sklearn.metrics', fromlist=['confusion_matrix']).confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title(f'Mango Grading (acc={acc:.2%})')
plt.tight_layout()
plt.savefig(f"models/{FRUIT}_confusion_matrix.png", dpi=150)

# Save to production
for d in ["models", os.path.join("app","ai","models")]:
    os.makedirs(d, exist_ok=True)
    shutil.copy2(TEMP_PATH, os.path.join(d, f"{FRUIT}_quality_ft.h5"))
    logger.info(f"Saved to {d}/{FRUIT}_quality_ft.h5")

logger.info("DONE")
