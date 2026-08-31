import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

JSON_DIR = os.path.join(
    PROJECT_ROOT,
    "json_data"
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

METRICS_DIR = os.path.join(
    PROJECT_ROOT,
    "metrics"
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

os.makedirs(
    METRICS_DIR,
    exist_ok=True
)


TRAIN_JSON = os.path.join(
    JSON_DIR,
    "train_dataset.json"
)

VALIDATION_JSON = os.path.join(
    JSON_DIR,
    "validation_dataset.json"
)

CLASSES_JSON = os.path.join(
    JSON_DIR,
    "classes.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (
    224,
    224
)

BATCH_SIZE = 16

EPOCHS = 20

NUM_CLASSES = 34


# ============================================================
# LOAD CLASS INFORMATION
# ============================================================

def load_classes():

    print("\nLoading class information...")

    with open(
            CLASSES_JSON,
            "r",
            encoding="utf-8"
    ) as file:

        class_data = json.load(file)

    if isinstance(class_data, dict) and "classes" in class_data:

        classes_data = class_data["classes"]

    else:

        classes_data = class_data

    if isinstance(classes_data, dict):

        if all(
                isinstance(value, int)
                for value in classes_data.values()
        ):

            classes = [
                class_name
                for class_name, class_index in sorted(
                    classes_data.items(),
                    key=lambda item: item[1]
                )
            ]

        else:

            classes = [
                class_name
                for class_index, class_name in sorted(
                    classes_data.items(),
                    key=lambda item: int(item[0])
                )
            ]

    elif isinstance(classes_data, list):

        classes = classes_data

    else:

        raise ValueError(
            "Unsupported classes.json structure"
        )

    print(
        f"Total Classes Loaded: {len(classes)}"
    )

    return classes


# ============================================================
# LOAD JSON RECORDS
# ============================================================

def load_json_records(
        json_path
):

    with open(
            json_path,
            "r",
            encoding="utf-8"
    ) as file:

        records = json.load(file)

    return records


# ============================================================
# CREATE DATASET
# ============================================================

def load_and_preprocess_image(
        image_path,
        label
):

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
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = preprocess_input(
        image
    )

    return image, label


def create_dataset(
        records,
        is_training=False
):

    image_paths = []
    labels = []

    for record in records:

        relative_path = record[
            "image_path"
        ]

        full_path = os.path.join(
            PROJECT_ROOT,
            relative_path
        )

        full_path = os.path.normpath(
            full_path
        )

        image_paths.append(
            full_path
        )

        labels.append(
            record[
                "class_index"
            ]
        )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            labels
        )
    )

    if is_training:

        dataset = dataset.shuffle(
            buffer_size=len(
                image_paths
            )
        )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# BUILD VGG16 MODEL
# ============================================================

def build_vgg16_model():

    print("\n" + "=" * 70)
    print("BUILDING VGG16 TRANSFER LEARNING MODEL")
    print("=" * 70)

    base_model = VGG16(

        weights="imagenet",

        include_top=False,

        input_shape=(
            224,
            224,
            3
        )

    )

    # Freeze pretrained VGG16 layers
    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=(
            224,
            224,
            3
        )
    )

    x = base_model(
        inputs,
        training=False
    )

    x = GlobalAveragePooling2D()(
        x
    )

    x = BatchNormalization()(
        x
    )

    x = Dense(
        512,
        activation="relu"
    )(x)

    x = Dropout(
        0.5
    )(x)

    x = Dense(
        256,
        activation="relu"
    )(x)

    x = Dropout(
        0.3
    )(x)

    outputs = Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = Model(
        inputs,
        outputs
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )

    print("\nVGG16 Model Created Successfully")

    print("\nModel Summary:")

    model.summary()

    return model


# ============================================================
# CALLBACKS
# ============================================================

def create_callbacks():

    best_model_path = os.path.join(
        MODELS_DIR,
        "best_food_vgg16.keras"
    )

    callbacks = [

        ModelCheckpoint(

            filepath=best_model_path,

            monitor="val_accuracy",

            mode="max",

            save_best_only=True,

            verbose=1

        ),

        EarlyStopping(

            monitor="val_accuracy",

            patience=5,

            mode="max",

            restore_best_weights=True,

            verbose=1

        ),

        ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=2,

            min_lr=0.000001,

            verbose=1

        )

    ]

    return callbacks


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_training_history(
        history
):

    history_data = {}

    for key, values in history.history.items():

        history_data[key] = [

            float(value)

            for value in values

        ]

    history_path = os.path.join(
        METRICS_DIR,
        "vgg16_training_history.json"
    )

    with open(
            history_path,
            "w",
            encoding="utf-8"
    ) as file:

        json.dump(
            history_data,
            file,
            indent=4
        )

    print(
        "\nTraining History Saved:"
    )

    print(
        history_path
    )


# ============================================================
# SAVE GRAPHS
# ============================================================

def save_training_graphs(
        history
):

    # ----------------------------
    # Accuracy Graph
    # ----------------------------

    plt.figure(
        figsize=(
            10,
            6
        )
    )

    plt.plot(
        history.history[
            "accuracy"
        ],
        label="Training Accuracy"
    )

    plt.plot(
        history.history[
            "val_accuracy"
        ],
        label="Validation Accuracy"
    )

    plt.title(
        "VGG16 Training vs Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.legend()

    plt.grid(
        True
    )

    accuracy_path = os.path.join(
        METRICS_DIR,
        "vgg16_accuracy_graph.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ----------------------------
    # Loss Graph
    # ----------------------------

    plt.figure(
        figsize=(
            10,
            6
        )
    )

    plt.plot(
        history.history[
            "loss"
        ],
        label="Training Loss"
    )

    plt.plot(
        history.history[
            "val_loss"
        ],
        label="Validation Loss"
    )

    plt.title(
        "VGG16 Training vs Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.legend()

    plt.grid(
        True
    )

    loss_path = os.path.join(
        METRICS_DIR,
        "vgg16_loss_graph.png"
    )

    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nAccuracy Graph Saved:"
    )

    print(
        accuracy_path
    )

    print(
        "\nLoss Graph Saved:"
    )

    print(
        loss_path
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "FOOD CLASSIFICATION USING VGG16"
    )

    print(
        "=" * 70
    )

    # Load classes
    classes = load_classes()

    print(
        "\nLoading Training JSON..."
    )

    train_records = load_json_records(
        TRAIN_JSON
    )

    print(
        f"Training Records: "
        f"{len(train_records)}"
    )

    print(
        "\nLoading Validation JSON..."
    )

    validation_records = load_json_records(
        VALIDATION_JSON
    )

    print(
        f"Validation Records: "
        f"{len(validation_records)}"
    )

    # Create datasets
    print(
        "\nCreating Training Dataset..."
    )

    train_dataset = create_dataset(
        train_records,
        is_training=True
    )

    print(
        "Training Dataset Ready"
    )

    print(
        "\nCreating Validation Dataset..."
    )

    validation_dataset = create_dataset(
        validation_records,
        is_training=False
    )

    print(
        "Validation Dataset Ready"
    )

    # Build model
    model = build_vgg16_model()

    # Create callbacks
    callbacks = create_callbacks()

    print("\n" + "=" * 70)
    print("STARTING VGG16 TRAINING")
    print("=" * 70)

    history = model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=EPOCHS,

        callbacks=callbacks

    )

    # Save final model
    final_model_path = os.path.join(
        MODELS_DIR,
        "food_classification_vgg16.keras"
    )

    model.save(
        final_model_path
    )

    print(
        "\nFinal VGG16 Model Saved:"
    )

    print(
        final_model_path
    )

    # Save history
    save_training_history(
        history
    )

    # Save graphs
    save_training_graphs(
        history
    )

    print("\n" + "=" * 70)
    print(
        "VGG16 TRAINING COMPLETED SUCCESSFULLY!"
    )
    print("=" * 70)

    print(
        f"\nTotal Classes: "
        f"{NUM_CLASSES}"
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()