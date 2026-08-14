"""
HarvestLenz – Fruit Type Classifier Training
Model : MobileNetV2 (transfer learning + fine-tuning)
Classes: grapes | mango | pineapple | pomegranate
Output : models/fruit_classifier.h5  +  models/class_indices.json
"""

import os, sys, json, shutil, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BACKEND = r'c:\Users\ranja\HarvestLenz\backend\backend'
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH       = 32
EPOCHS_HEAD = 10       # Train only the new head first
EPOCHS_FT   = 15       # Then fine-tune top layers
LR_HEAD     = 1e-3
LR_FT       = 1e-5
CLASSES     = ['grapes', 'mango', 'pineapple', 'pomegranate']
SPLIT_DIR   = os.path.join(BACKEND, 'fruit_split_cls')   # working split

# ── Step 1: Build a balanced dataset split ────────────────────────────────────
SOURCES = {
    'grapes':       r'c:\Users\ranja\HarvestLenz\dataset-grapes\dataset-grapes',
    'mango':        r'c:\Users\ranja\HarvestLenz\dataset_mango\dataset_mango',
    'pineapple':    r'c:\Users\ranja\HarvestLenz\dataset-pineapple\dataset-pineapple',
    'pomegranate':  r'c:\Users\ranja\HarvestLenz\dataset-pomegranate\dataset-pomegranate',
}

# Re-create split directory
if os.path.exists(SPLIT_DIR):
    shutil.rmtree(SPLIT_DIR)

TRAIN_CAP = 500   # max per class
VAL_CAP   = 100

for cls in CLASSES:
    src_root = SOURCES[cls]
    imgs = []
    for sub in os.listdir(src_root):
        sub_path = os.path.join(src_root, sub)
        if os.path.isdir(sub_path):
            imgs += [os.path.join(sub_path, f) for f in os.listdir(sub_path)
                     if f.lower().endswith(('.jpg','.jpeg','.png'))]
        elif sub.lower().endswith(('.jpg','.jpeg','.png')):
            imgs.append(os.path.join(src_root, sub))

    random.seed(42)
    random.shuffle(imgs)

    val_imgs   = imgs[:VAL_CAP]
    train_imgs = imgs[VAL_CAP : VAL_CAP + TRAIN_CAP]

    for split, split_imgs in [('train', train_imgs), ('val', val_imgs)]:
        dest = os.path.join(SPLIT_DIR, split, cls)
        os.makedirs(dest, exist_ok=True)
        for p in split_imgs:
            shutil.copy2(p, dest)

    print(f"  {cls:15s}: {len(train_imgs)} train  |  {len(val_imgs)} val")

print("\nDataset split ready.\n")

# ── Step 2: Data generators ───────────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.75, 1.25],
    fill_mode='nearest',
)
val_gen = ImageDataGenerator(rescale=1./255)

train_ds = train_gen.flow_from_directory(
    os.path.join(SPLIT_DIR, 'train'),
    target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', shuffle=True,
    classes=CLASSES,
)
val_ds = val_gen.flow_from_directory(
    os.path.join(SPLIT_DIR, 'val'),
    target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', shuffle=False,
    classes=CLASSES,
)

print("Class indices:", train_ds.class_indices)

# ── Step 3: Build model ───────────────────────────────────────────────────────
base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
base.trainable = False   # freeze base for head training

inputs  = keras.Input(shape=(*IMG_SIZE, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(CLASSES), activation='softmax')(x)
model   = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(LR_HEAD),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary(line_length=80)

# ── Step 4: Train head ────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"Phase 1 – Head training  ({EPOCHS_HEAD} epochs, lr={LR_HEAD})")
print("="*60)

cb_head = [
    keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
]
hist1 = model.fit(train_ds, epochs=EPOCHS_HEAD, validation_data=val_ds, callbacks=cb_head)

print(f"\nBest head val_acc: {max(hist1.history['val_accuracy']):.4f}")

# ── Step 5: Fine-tune top 50 layers ──────────────────────────────────────────
print("\n" + "="*60)
print(f"Phase 2 – Fine-tuning  ({EPOCHS_FT} epochs, lr={LR_FT})")
print("="*60)

base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(LR_FT),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

cb_ft = [
    keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint(
        'models/fruit_classifier.h5', save_best_only=True,
        monitor='val_accuracy', verbose=1,
    ),
]
hist2 = model.fit(train_ds, epochs=EPOCHS_FT, validation_data=val_ds, callbacks=cb_ft)

# ── Step 6: Final evaluation ──────────────────────────────────────────────────
print("\n" + "="*60)
print("Final Evaluation on Validation Set")
print("="*60)

loss, acc = model.evaluate(val_ds, verbose=0)
print(f"  Val Loss     : {loss:.4f}")
print(f"  Val Accuracy : {acc*100:.2f}%")

# Per-class accuracy
val_ds.reset()
preds  = model.predict(val_ds, verbose=0)
y_pred = np.argmax(preds, axis=1)
y_true = val_ds.classes

from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# ── Step 7: Save class indices ────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
class_indices = {cls: i for i, cls in enumerate(CLASSES)}
with open('models/class_indices.json', 'w') as f:
    json.dump(class_indices, f, indent=2)
print("\nSaved: models/fruit_classifier.h5")
print("Saved: models/class_indices.json")
print("\nTraining complete!")
