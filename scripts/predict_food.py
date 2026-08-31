import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

JSON_DIR = os.path.join(
    PROJECT_ROOT,
    "json_data"
)


MODEL_PATH = os.path.join(
    MODELS_DIR,
    "best_food_resnet50.keras"
)

CLASSES_PATH = os.path.join(
    JSON_DIR,
    "classes.json"
)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_classes():

    print("\nLoading class information...")

    if not os.path.exists(CLASSES_PATH):

        raise FileNotFoundError(
            f"Classes file not found:\n{CLASSES_PATH}"
        )

    with open(
            CLASSES_PATH,
            "r",
            encoding="utf-8"
    ) as file:

        classes_data = json.load(file)

    # Handle project classes.json structure
    if (
        isinstance(classes_data, dict)
        and "classes" in classes_data
    ):

        classes_data = classes_data["classes"]

    classes = []

    if isinstance(classes_data, list):

        # Handle:
        # [{"class_name": "...", "class_index": 0}, ...]
        if (
            len(classes_data) > 0
            and isinstance(classes_data[0], dict)
        ):

            sorted_classes = sorted(
                classes_data,
                key=lambda item: item["class_index"]
            )

            classes = [
                item["class_name"]
                for item in sorted_classes
            ]

        else:

            classes = classes_data

    elif isinstance(classes_data, dict):

        # Handle {"apple_pie": 0, ...}
        if all(
                isinstance(value, int)
                for value in classes_data.values()
        ):

            sorted_items = sorted(
                classes_data.items(),
                key=lambda item: item[1]
            )

            classes = [
                item[0]
                for item in sorted_items
            ]

        # Handle {"0": "apple_pie", ...}
        else:

            sorted_items = sorted(
                classes_data.items(),
                key=lambda item: int(item[0])
            )

            classes = [
                item[1]
                for item in sorted_items
            ]

    else:

        raise ValueError(
            "Unsupported classes.json structure"
        )

    print(
        f"Total Classes Loaded: {len(classes)}"
    )

    return classes


# ============================================================
# LOAD TRAINED RESNET50 MODEL
# ============================================================

def load_food_model():

    print("\nLoading trained ResNet50 model...")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "ResNet50 Model Loaded Successfully"
    )

    print(
        f"Model Input Shape: {model.input_shape}"
    )

    print(
        f"Model Output Shape: {model.output_shape}"
    )

    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
        image_path
):

    print(
        "\nPreprocessing image..."
    )

    if not os.path.exists(
            image_path
    ):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(
            224,
            224
        )
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # IMPORTANT:
    # Same preprocessing used during ResNet50 training
    image_array = preprocess_input(
        image_array
    )

    return image_array


# ============================================================
# PREDICT FOOD
# ============================================================

def predict_food(
        model,
        image_array,
        classes,
        top_k=3
):

    print(
        "\nGenerating prediction..."
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    top_indices = np.argsort(
        predictions
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append(
            {
                "class_name":
                    classes[index],

                "confidence":
                    float(
                        predictions[index]
                        * 100
                    )
            }
        )

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
        results
):

    print("\n" + "=" * 70)

    print(
        "FOOD CLASSIFICATION RESULT"
    )

    print("=" * 70)

    best_prediction = results[0]

    print(
        f"\nPredicted Food: "
        f"{best_prediction['class_name']}"
    )

    print(
        f"Confidence: "
        f"{best_prediction['confidence']:.2f}%"
    )

    print(
        "\nTOP 3 PREDICTIONS"
    )

    print("-" * 70)

    for rank, result in enumerate(
            results,
            start=1
    ):

        print(
            f"{rank}. "
            f"{result['class_name']:<25} "
            f"{result['confidence']:.2f}%"
        )

    print("=" * 70)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "FOOD CLASSIFICATION - SINGLE IMAGE PREDICTION"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # LOAD CLASSES
    # --------------------------------------------------------

    classes = load_classes()


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_food_model()


    # --------------------------------------------------------
    # ENTER IMAGE PATH
    # --------------------------------------------------------

    print(
        "\nEnter the complete path of a food image."
    )

    image_path = input(
        "\nImage Path: "
    ).strip()

    # Remove quotes if pasted from Windows
    image_path = image_path.strip(
        '"'
    )

    image_path = image_path.strip(
        "'"
    )


    # --------------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------------

    image_array = preprocess_image(
        image_path
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    results = predict_food(
        model,
        image_array,
        classes,
        top_k=3
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    display_results(
        results
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()