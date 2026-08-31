import json
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_JSON = (
    PROJECT_ROOT
    / "json_data"
    / "train_dataset.json"
)

IMG_HEIGHT = 128
IMG_WIDTH = 128


print("=" * 60)
print("DATASET LOADING TEST")
print("=" * 60)


# Load JSON
with open(
    TRAIN_JSON,
    "r",
    encoding="utf-8"
) as file:

    train_data = json.load(file)


print(f"\nTotal JSON Records: {len(train_data)}")


# Test first 10 images
TEST_COUNT = 10


images = []
labels = []


for index, record in enumerate(
    train_data[:TEST_COUNT],
    start=1
):

    image_path = (
        PROJECT_ROOT
        / Path(record["image_path"])
    )

    print(f"\nImage {index}")
    print(f"Class: {record['class_name']}")
    print(f"Label: {record['class_index']}")
    print(f"Path: {image_path}")

    # Check file exists
    if not image_path.exists():

        raise FileNotFoundError(
            f"\nIMAGE NOT FOUND:\n{image_path}"
        )

    # Open and preprocess image
    with Image.open(image_path) as image:

        image = image.convert("RGB")

        image = image.resize(
            (IMG_WIDTH, IMG_HEIGHT)
        )

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = image_array / 255.0

        images.append(image_array)

        labels.append(
            record["class_index"]
        )


X_test = np.array(
    images,
    dtype=np.float32
)

y_test = np.array(
    labels,
    dtype=np.int32
)


print("\n" + "=" * 60)
print("TEST RESULT")
print("=" * 60)

print(f"\nImages Loaded Successfully: {len(X_test)}")

print(
    f"Image Array Shape: "
    f"{X_test.shape}"
)

print(
    f"Label Array Shape: "
    f"{y_test.shape}"
)

print(
    f"Labels: {y_test}"
)

print("\nSUCCESS! DATASET PATHS AND IMAGES ARE WORKING.")

print("=" * 60)