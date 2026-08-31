import json
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

JSON_DIR = PROJECT_ROOT / "json_data"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_JSON = JSON_DIR / "train_dataset.json"
VALIDATION_JSON = JSON_DIR / "validation_dataset.json"
CLASSES_JSON = JSON_DIR / "classes.json"


# ============================================================
# CONFIGURATION
# ============================================================

IMG_HEIGHT = 128
IMG_WIDTH = 128

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# LOAD JSON DATA
# ============================================================

print("\n" + "=" * 70)
print("FOOD CLASSIFICATION USING CUSTOM CNN")
print("=" * 70)

print("\nLoading JSON dataset files...")

with open(TRAIN_JSON, "r", encoding="utf-8") as file:
    train_data = json.load(file)

with open(VALIDATION_JSON, "r", encoding="utf-8") as file:
    validation_data = json.load(file)

with open(CLASSES_JSON, "r", encoding="utf-8") as file:
    classes_data = json.load(file)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = [
    item["class_name"]
    for item in classes_data["classes"]
]

NUM_CLASSES = len(class_names)

print(f"\nTotal Classes: {NUM_CLASSES}")
print(f"Training Records: {len(train_data)}")
print(f"Validation Records: {len(validation_data)}")


# ============================================================
# CREATE PATHS AND LABELS
# ============================================================

def extract_paths_and_labels(records):

    image_paths = []
    labels = []

    for record in records:

        full_path = (
            PROJECT_ROOT
            / Path(record["image_path"])
        )

        image_paths.append(
            str(full_path)
        )

        labels.append(
            record["class_index"]
        )

    return image_paths, labels


train_paths, train_labels = extract_paths_and_labels(
    train_data
)

validation_paths, validation_labels = (
    extract_paths_and_labels(
        validation_data
    )
)


# ============================================================
# IMAGE PREPROCESSING FUNCTION
# ============================================================

def preprocess_image(image_path, label):

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        [IMG_HEIGHT, IMG_WIDTH]
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = image / 255.0

    return image, label


# ============================================================
# CREATE TF.DATA DATASETS
# ============================================================

print("\nCreating TensorFlow data pipeline...")


train_dataset = tf.data.Dataset.from_tensor_slices(

    (
        train_paths,
        train_labels
    )

)

train_dataset = train_dataset.shuffle(
    buffer_size=len(train_paths),
    reshuffle_each_iteration=True
)

train_dataset = train_dataset.map(
    preprocess_image,
    num_parallel_calls=AUTOTUNE
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)


validation_dataset = tf.data.Dataset.from_tensor_slices(

    (
        validation_paths,
        validation_labels
    )

)

validation_dataset = validation_dataset.map(
    preprocess_image,
    num_parallel_calls=AUTOTUNE
)

validation_dataset = validation_dataset.batch(
    BATCH_SIZE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)


print("TensorFlow data pipeline created successfully.")


# ============================================================
# DATASET CHECK
# ============================================================

print("\nChecking one training batch...")

for batch_images, batch_labels in train_dataset.take(1):

    print(
        f"Batch Image Shape: "
        f"{batch_images.shape}"
    )

    print(
        f"Batch Label Shape: "
        f"{batch_labels.shape}"
    )


# ============================================================
# BUILD CUSTOM CNN MODEL
# ============================================================

print("\nBuilding Custom CNN Model...")


model = models.Sequential([

    layers.Input(
        shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            3
        )
    ),

    # --------------------------------------------------------
    # BLOCK 1
    # --------------------------------------------------------

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # --------------------------------------------------------
    # BLOCK 2
    # --------------------------------------------------------

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # --------------------------------------------------------
    # BLOCK 3
    # --------------------------------------------------------

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # --------------------------------------------------------
    # BLOCK 4
    # --------------------------------------------------------

    layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        512,
        activation="relu"
    ),

    layers.Dropout(
        0.5
    ),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )

])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CUSTOM CNN MODEL ARCHITECTURE")
print("=" * 70)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

BEST_MODEL_PATH = (
    MODELS_DIR
    / "best_food_cnn.keras"
)

FINAL_MODEL_PATH = (
    MODELS_DIR
    / "food_classification_cnn.keras"
)


checkpoint = ModelCheckpoint(

    filepath=str(BEST_MODEL_PATH),

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)


early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=7,

    restore_best_weights=True,

    verbose=1

)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=0.000001,

    verbose=1

)


callbacks = [

    checkpoint,
    early_stopping,
    reduce_lr

]


# ============================================================
# START TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING CUSTOM CNN TRAINING")
print("=" * 70)

print(f"\nEpochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(
    f"Image Size: "
    f"{IMG_HEIGHT} x {IMG_WIDTH}"
)


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1

)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    str(FINAL_MODEL_PATH)
)

print(
    "\nFinal CNN model saved successfully."
)


# ============================================================
# SAVE TRAINING HISTORY AS JSON
# ============================================================

history_data = {}

for key, values in history.history.items():

    history_data[key] = [
        float(value)
        for value in values
    ]


HISTORY_PATH = (
    METRICS_DIR
    / "cnn_training_history.json"
)


with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        history_data,
        file,
        indent=4
    )


print(
    "Training history saved successfully."
)


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Custom CNN Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid()


ACCURACY_GRAPH = (
    METRICS_DIR
    / "cnn_accuracy_graph.png"
)


plt.savefig(
    ACCURACY_GRAPH
)

plt.close()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Custom CNN Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid()


LOSS_GRAPH = (
    METRICS_DIR
    / "cnn_loss_graph.png"
)


plt.savefig(
    LOSS_GRAPH
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("CUSTOM CNN TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)

print(f"\nBest Model:\n{BEST_MODEL_PATH}")

print(f"\nFinal Model:\n{FINAL_MODEL_PATH}")

print(f"\nTraining History:\n{HISTORY_PATH}")

print(f"\nAccuracy Graph:\n{ACCURACY_GRAPH}")

print(f"\nLoss Graph:\n{LOSS_GRAPH}")

print(f"\nTotal Classes: {NUM_CLASSES}")

print("\n" + "=" * 70)