"""
Shared CNN Architectures for HarvestLenz Grading

Fruits:       MobileNetV2, 3-class (Better / Good / Reject)
Vegetables:   EfficientNetB0, 2-class (Good / Reject)

Both build functions expose the pretrained base as `model._base_model`
so the training pipeline can fine-tune its top layers.
"""
import numpy as np
from typing import Tuple, Dict

SUPPORTED_FRUITS = [
    "mango", "pineapple", "grapes", "pomegranate",
    "orange", "guava", "kiwi", "watermelon", "banana",
    "cocoa", "coffee", "strawberry", "plum", "peach", "pear",
]
SUPPORTED_VEGETABLES = [
    "carrot", "tomato", "onion", "cucumber", "capsicum", "potato",
]

FRUIT_CLASSES = ["Better", "Good", "Reject"]

FRUIT_NUM_CLASSES: Dict[str, int] = {f: 3 for f in SUPPORTED_FRUITS}
FRUIT_NUM_CLASSES.update({v: 2 for v in SUPPORTED_VEGETABLES})

VEGETABLE_BACKBONES: Dict[str, str] = {v: "efficientnet" for v in SUPPORTED_VEGETABLES}

_FRUIT_GRADE_MAP = {0: "Better", 1: "Good", 2: "Reject"}


def build_grading_model(num_classes: int = 3, input_size: Tuple[int, int] = (224, 224)):
    """
    Builds a MobileNetV2 transfer-learning model for fruit quality grading.

    Architecture:
        MobileNetV2 (ImageNet weights, frozen base)
        -> GlobalAveragePooling2D
        -> Dense(256, activation='relu')
        -> BatchNormalization
        -> Dropout(0.4)
        -> Dense(128, activation='relu')
        -> Dropout(0.3)
        -> Dense(num_classes, activation='softmax')

    Returns a compiled Keras model. The pretrained base is exposed as
    `model._base_model` so training can fine-tune its top layers.
    """
    from keras.applications import MobileNetV2
    from keras.layers import (
        GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
    )
    from keras.models import Model

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*input_size, 3),
        input_tensor=Input(shape=(*input_size, 3))
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model._base_model = base_model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_vegetable_model(num_classes: int = 2, input_size: Tuple[int, int] = (224, 224)):
    """
    Builds an EfficientNetB0 transfer-learning model for vegetable grading.

    Architecture:
        EfficientNetB0 (ImageNet weights, frozen base)
        -> GlobalAveragePooling2D
        -> Dense(256, activation='relu')
        -> BatchNormalization
        -> Dropout(0.4)
        -> Dense(128, activation='relu')
        -> Dropout(0.3)
        -> Dense(num_classes, activation='softmax')

    Returns a compiled Keras model. The pretrained base is exposed as
    `model._base_model` so training can fine-tune its top layers.
    """
    from keras.applications import EfficientNetB0
    from keras.layers import (
        GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
    )
    from keras.models import Model

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*input_size, 3),
        input_tensor=Input(shape=(*input_size, 3))
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model._base_model = base_model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def preprocess_image(img_bgr: np.ndarray, backbone: str = "mobilenet") -> np.ndarray:
    """
    Prepares a BGR OpenCV image for CNN inference, matching the exact
    preprocessing used during training (Keras `preprocess_input`).

    Steps:
        1. Resize to 224x224
        2. Convert BGR -> RGB
        3. Backbone-specific `preprocess_input` scaling
    """
    import cv2

    img = cv2.resize(img_bgr, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if backbone == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input as fn
    else:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as fn
    return fn(img_rgb.astype("float32"))


def decode_prediction(probs: np.ndarray, fruit_key: str = "") -> Tuple[str, float, str]:
    """
    Decodes softmax probabilities into (grade, confidence, disease_name).

    Class mapping (3-class fruits):
        0 -> Better, 1 -> Good, 2 -> Reject

    Class mapping (2-class vegetables):
        0 -> Good, 1 -> Reject

    `disease_name` is returned empty for generic grading models.
    """
    nc = FRUIT_NUM_CLASSES.get(fruit_key, 3)

    if nc == 2:
        idx = int(np.argmax(probs))
        grade = "Good" if idx == 0 else "Reject"
        confidence = float(probs[idx])
        return grade, confidence, ""

    idx = int(np.argmax(probs))
    grade = _FRUIT_GRADE_MAP.get(idx, "Good")
    confidence = float(probs[idx])
    return grade, confidence, ""
