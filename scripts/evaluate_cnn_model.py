import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

JSON_DIR = os.path.join(PROJECT_ROOT, "json_data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
METRICS_DIR = os.path.join(PROJECT_ROOT, "metrics")

VALIDATION_JSON = os.path.join(
    JSON_DIR,
    "validation_dataset.json"
)

CLASSES_JSON = os.path.join(
    JSON_DIR,
    "classes.json"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "best_food_cnn.keras"
)

os.makedirs(METRICS_DIR, exist_ok=True)


# ============================================================
# LOAD CLASSES
# ============================================================

def load_classes():

    print("\nLoading class information...")

    with open(
            CLASSES_JSON,
            "r",
            encoding="utf-8"
    ) as file:

        class_data = json.load(file)

    print("\nClasses JSON Type:", type(class_data))

    # --------------------------------------------------------
    # CASE 1: Dictionary containing "classes"
    # --------------------------------------------------------

    if isinstance(class_data, dict) and "classes" in class_data:

        classes_data = class_data["classes"]

    else:

        classes_data = class_data

    # --------------------------------------------------------
    # CASE 2: List
    # --------------------------------------------------------

    if isinstance(classes_data, list):

        classes = []

        for item in classes_data:

            if isinstance(item, str):

                classes.append(item)

            elif isinstance(item, dict):

                # Try common keys
                if "class_name" in item:

                    classes.append(
                        item["class_name"]
                    )

                elif "name" in item:

                    classes.append(
                        item["name"]
                    )

                else:

                    print(
                        "Unexpected class dictionary:",
                        item
                    )

                    raise ValueError(
                        "Could not find class name"
                    )

    # --------------------------------------------------------
    # CASE 3: Dictionary
    # --------------------------------------------------------

    elif isinstance(classes_data, dict):

        # Example:
        # {"apple_pie": 0, "burger": 1}

        if all(
                isinstance(value, int)
                for value in classes_data.values()
        ):

            classes = [

                class_name

                for class_name, class_index

                in sorted(
                    classes_data.items(),
                    key=lambda item: item[1]
                )

            ]

        # Example:
        # {"0": "apple_pie", "1": "burger"}

        elif all(
                isinstance(value, str)
                for value in classes_data.values()
        ):

            classes = [

                class_name

                for class_index, class_name

                in sorted(
                    classes_data.items(),
                    key=lambda item: int(item[0])
                )

            ]

        else:

            print(
                "\nUnexpected classes.json structure:"
            )

            print(
                json.dumps(
                    classes_data,
                    indent=4
                )
            )

            raise ValueError(
                "Unsupported classes.json structure"
            )

    else:

        raise ValueError(
            "Unsupported classes.json format"
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print(
        f"\nTotal Classes Loaded: "
        f"{len(classes)}"
    )

    print(
        "\nFirst 10 Classes:"
    )

    for index, class_name in enumerate(
            classes[:10]
    ):

        print(
            f"{index}: {class_name}"
        )

    return classes


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

def load_validation_data():

    print("\nLoading validation dataset records...")

    with open(VALIDATION_JSON, "r", encoding="utf-8") as file:
        records = json.load(file)

    print(f"Total Validation Records: {len(records)}")

    images = []
    labels = []

    print("\nLoading validation images...")

    for index, record in enumerate(records):

        relative_path = record["image_path"]

        image_path = os.path.join(
            PROJECT_ROOT,
            relative_path
        )

        image_path = os.path.normpath(image_path)

        image = tf.keras.utils.load_img(
            image_path,
            target_size=(128, 128)
        )

        image_array = tf.keras.utils.img_to_array(image)

        images.append(image_array)

        labels.append(record["class_index"])

        if (index + 1) % 100 == 0:
            print(
                f"Loaded {index + 1}/{len(records)} validation images"
            )

    images = np.array(
        images,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.int32
    )

    # Normalize images
    images = images / 255.0

    print("\nValidation Dataset Loaded Successfully")
    print("Image Shape:", images.shape)
    print("Label Shape:", labels.shape)

    return images, labels


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    print("\nLoading trained CNN model...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model Loaded Successfully")

    return model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
        model,
        validation_images,
        validation_labels
):

    print("\n" + "=" * 70)
    print("EVALUATING CNN MODEL")
    print("=" * 70)

    loss, accuracy = model.evaluate(
        validation_images,
        validation_labels,
        verbose=1
    )

    print("\nFinal Validation Loss:", loss)
    print("Final Validation Accuracy:", accuracy)

    return loss, accuracy


# ============================================================
# PREDICTIONS
# ============================================================

def generate_predictions(
        model,
        validation_images
):

    print("\nGenerating predictions...")

    predictions = model.predict(
        validation_images,
        verbose=1
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    print("Predictions Generated Successfully")

    return predicted_labels, predictions


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def create_classification_report(
        true_labels,
        predicted_labels,
        classes
):

    print("\nGenerating Classification Report...")

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0
    )

    report_path = os.path.join(
        METRICS_DIR,
        "classification_report.txt"
    )

    with open(
            report_path,
            "w",
            encoding="utf-8"
    ) as file:

        file.write(report)

    print("\nClassification Report:")
    print(report)

    print(
        f"\nClassification Report Saved:\n{report_path}"
    )

    return report

    # --------------------------------------------------------
    # Create Classification Report
    # --------------------------------------------------------

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    # --------------------------------------------------------
    # Save Report
    # --------------------------------------------------------

    report_path = os.path.join(
        METRICS_DIR,
        "classification_report.txt"
    )

    with open(
            report_path,
            "w",
            encoding="utf-8"
    ) as file:

        file.write(report)

    print("\nClassification Report:")
    print(report)

    print(
        f"\nClassification Report Saved:\n{report_path}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
        true_labels,
        predicted_labels,
        classes
):

    print("\nCreating Confusion Matrix...")

    matrix = confusion_matrix(
        true_labels,
        predicted_labels
    )

    plt.figure(
        figsize=(20, 18)
    )

    plt.imshow(matrix)

    plt.title(
        "Food Classification Confusion Matrix"
    )

    plt.colorbar()

    tick_positions = np.arange(
        len(classes)
    )

    plt.xticks(
        tick_positions,
        classes,
        rotation=90
    )

    plt.yticks(
        tick_positions,
        classes
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.tight_layout()

    matrix_path = os.path.join(
        METRICS_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        matrix_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion Matrix Saved:\n{matrix_path}"
    )

    return matrix


# ============================================================
# PER CLASS ACCURACY
# ============================================================

def calculate_per_class_accuracy(
        matrix,
        classes
):

    print("\n" + "=" * 70)
    print("PER CLASS ACCURACY")
    print("=" * 70)

    class_results = []

    for index, class_name in enumerate(classes):

        total_images = np.sum(
            matrix[index]
        )

        correct_predictions = matrix[
            index,
            index
        ]

        if total_images > 0:

            accuracy = (
                correct_predictions /
                total_images
            ) * 100

        else:
            accuracy = 0.0

        result = {
            "class_name": class_name,
            "total_validation_images": int(
                total_images
            ),
            "correct_predictions": int(
                correct_predictions
            ),
            "accuracy_percent": round(
                float(accuracy),
                2
            )
        }

        class_results.append(
            result
        )

        print(
            f"{class_name:<25} "
            f"{accuracy:.2f}%"
        )

    # Sort from lowest accuracy to highest
    sorted_results = sorted(
        class_results,
        key=lambda x: x[
            "accuracy_percent"
        ]
    )

    results_path = os.path.join(
        METRICS_DIR,
        "per_class_accuracy.json"
    )

    with open(
            results_path,
            "w",
            encoding="utf-8"
    ) as file:

        json.dump(
            sorted_results,
            file,
            indent=4
        )

    print("\nPer Class Accuracy Saved:")
    print(results_path)

    return sorted_results


# ============================================================
# SAVE EVALUATION SUMMARY
# ============================================================

def save_evaluation_summary(
        validation_loss,
        validation_accuracy,
        per_class_results
):

    overall_accuracy = (
        validation_accuracy * 100
    )

    weakest_classes = (
        per_class_results[:5]
    )

    strongest_classes = (
        per_class_results[-5:]
    )

    summary = {

        "total_classes": 34,

        "validation_loss":
            round(
                float(validation_loss),
                6
            ),

        "validation_accuracy_percent":
            round(
                float(overall_accuracy),
                2
            ),

        "weakest_classes":
            weakest_classes,

        "strongest_classes":
            strongest_classes

    }

    summary_path = os.path.join(
        METRICS_DIR,
        "evaluation_summary.json"
    )

    with open(
            summary_path,
            "w",
            encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\nEvaluation Summary Saved:")
    print(summary_path)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("FOOD CLASSIFICATION CNN - MODEL EVALUATION")
    print("=" * 70)

    classes = load_classes()

    validation_images, validation_labels = (
        load_validation_data()
    )

    model = load_model()

    validation_loss, validation_accuracy = (
        evaluate_model(
            model,
            validation_images,
            validation_labels
        )
    )

    predicted_labels, predictions = (
        generate_predictions(
            model,
            validation_images
        )
    )

    create_classification_report(
        validation_labels,
        predicted_labels,
        classes
    )

    matrix = create_confusion_matrix(
        validation_labels,
        predicted_labels,
        classes
    )

    per_class_results = (
        calculate_per_class_accuracy(
            matrix,
            classes
        )
    )

    save_evaluation_summary(
        validation_loss,
        validation_accuracy,
        per_class_results
    )

    print("\n" + "=" * 70)
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    print(
        f"\nModel Accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )

    print(
        "\nAll evaluation files are available in:"
    )

    print(
        METRICS_DIR
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()