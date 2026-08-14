"""
Mango Color Augmentation Script
Generates green, reddish-yellow, and orange mango shade variations
from existing dataset images to improve model robustness across
different mango varieties (Alphonso, Dasheri, Langra, Totapuri, etc.)
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

DATASET_DIR = r"C:\Users\madha\OneDrive\Desktop\HarvestLenz\dataset_mango\dataset_mango"
OUTPUT_SUFFIX = "_shade"  # appended to filenames of augmented images

# Each shade profile shifts HSV channels to simulate different mango varieties
SHADE_PROFILES = {
    # Green mango (unripe / Dasheri / Langra)
    "green": {
        "hue_shift": 20,        # shift hue toward green
        "sat_scale": 0.85,      # slightly desaturate
        "val_scale": 0.90,      # slightly darker
        "brightness": 0.92,
        "contrast": 1.05,
    },
    # Reddish-yellow (Alphonso / Kesar ripe)
    "reddish_yellow": {
        "hue_shift": -10,       # shift hue toward red/orange
        "sat_scale": 1.15,      # more saturated
        "val_scale": 1.0,
        "brightness": 1.05,
        "contrast": 1.10,
    },
    # Deep orange (fully ripe / Banganapalli)
    "orange": {
        "hue_shift": -15,       # shift hue toward orange
        "sat_scale": 1.20,      # rich saturation
        "val_scale": 1.05,      # slightly brighter
        "brightness": 1.08,
        "contrast": 1.05,
    },
    # Pale yellow (Totapuri / light variety)
    "pale_yellow": {
        "hue_shift": 5,         # slight shift toward yellow-green
        "sat_scale": 0.70,      # less saturated
        "val_scale": 1.15,      # brighter
        "brightness": 1.12,
        "contrast": 0.95,
    },
    # Deep green (raw / unripe Alphonso)
    "deep_green": {
        "hue_shift": 30,
        "sat_scale": 0.90,
        "val_scale": 0.75,      # darker
        "brightness": 0.85,
        "contrast": 1.10,
    },
    # Red blush (apple mango / Tommy Atkins style)
    "red_blush": {
        "hue_shift": -20,
        "sat_scale": 1.10,
        "val_scale": 0.95,
        "brightness": 0.98,
        "contrast": 1.15,
    },
}


def apply_shade(img_bgr, profile):
    """Apply an HSV-based color shift + brightness/contrast adjustment."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # Shift hue (OpenCV H range is 0-180)
    h = (h + profile["hue_shift"]) % 180

    # Scale saturation
    s = np.clip(s * profile["sat_scale"], 0, 255)

    # Scale value (brightness)
    v = np.clip(v * profile["val_scale"], 0, 255)

    hsv[:,:,0] = h
    hsv[:,:,1] = s
    hsv[:,:,2] = v

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Additional brightness/contrast via PIL
    pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    pil_img = ImageEnhance.Brightness(pil_img).enhance(profile["brightness"])
    pil_img = ImageEnhance.Contrast(pil_img).enhance(profile["contrast"])

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def augment_dataset():
    total_generated = 0

    for grade in ["Good", "Better", "Reject"]:
        grade_dir = os.path.join(DATASET_DIR, grade)
        if not os.path.isdir(grade_dir):
            print(f"Skipping {grade}: directory not found")
            continue

        files = [f for f in os.listdir(grade_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        # Skip already-augmented files
        files = [f for f in files if OUTPUT_SUFFIX not in f]
        
        print(f"\n{grade}: {len(files)} original images")

        # Sample images to augment (use all for small sets, ~300 for large)
        sample_size = min(len(files), 300)
        sampled = random.sample(files, sample_size)

        grade_count = 0
        for fname in sampled:
            src_path = os.path.join(grade_dir, fname)
            img = cv2.imread(src_path)
            if img is None:
                continue

            # Pick 2-3 random shade profiles per image
            num_shades = random.randint(2, 3)
            chosen_shades = random.sample(list(SHADE_PROFILES.keys()), num_shades)

            for shade_name in chosen_shades:
                profile = SHADE_PROFILES[shade_name]
                augmented = apply_shade(img, profile)

                name_part = os.path.splitext(fname)[0]
                ext = os.path.splitext(fname)[1]
                out_name = f"{name_part}_{shade_name}{OUTPUT_SUFFIX}{ext}"
                out_path = os.path.join(grade_dir, out_name)

                cv2.imwrite(out_path, augmented)
                grade_count += 1

        total_generated += grade_count
        print(f"  Generated {grade_count} augmented images ({shade_name} variants)")

    print(f"\nTotal augmented images generated: {total_generated}")
    print("Run train_grading_model.py to retrain with expanded dataset.")


if __name__ == "__main__":
    random.seed(42)
    augment_dataset()
