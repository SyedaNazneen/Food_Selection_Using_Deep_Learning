import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import preprocess_input

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ============================================================
# PROJECT PATH CONFIGURATION
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
    "best_food_vgg16.keras"
)


os.makedirs(
    METRICS_DIR,
    exist_ok=True
)


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

        classes_data = json.load(file)

    print(
        f"\nClasses JSON Type: "
        f"{type(classes_data)}"
    )

    # ========================================================
    # GET CLASSES DATA
    # ========================================================

    if (
        isinstance(classes_data, dict)
        and "classes" in classes_data
    ):

        class_records = classes_data["classes"]

        # Sort according to class_index
        class_records = sorted(
            class_records,
            key=lambda x: x["class_index"]
        )

        classes = [
            record["class_name"]
            for record in class_records
        ]

    elif isinstance(classes_data, list):

        classes = classes_data

    else:

        raise ValueError(
            "\nUnsupported classes.json format."
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    if len(classes) != 34:

        raise ValueError(
            f"\nExpected 34 classes, "
            f"but found {len(classes)}"
        )

    print(
        f"\nTotal Classes Loaded: "
        f"{len(classes)}"
    )

    print("\nFirst 10 Classes:")

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

    print(
        "\nLoading validation dataset records..."
    )


    with open(
            VALIDATION_JSON,
            "r",
            encoding="utf-8"
    ) as file:

        records = json.load(file)


    print(
        f"Total Validation Records: "
        f"{len(records)}"
    )


    images = []

    labels = []


    print(
        "\nLoading validation images..."
    )


    for index, record in enumerate(
            records
    ):

        relative_path = record[
            "image_path"
        ]


        image_path = os.path.join(
            PROJECT_ROOT,
            relative_path
        )


        image_path = os.path.normpath(
            image_path
        )

        image = tf.keras.utils.load_img(
            image_path,
            target_size=(224, 224)
        )


        image_array = (
            tf.keras.utils.img_to_array(
                image
            )
        )


        images.append(
            image_array
        )


        labels.append(
            record[
                "class_index"
            ]
        )


        if (
                index + 1
        ) % 100 == 0:

            print(
                f"Loaded "
                f"{index + 1}/"
                f"{len(records)} "
                f"validation images"
            )


    images = np.array(
        images,
        dtype=np.float32
    )


    labels = np.array(
        labels,
        dtype=np.int32
    )

    # VGG16 preprocessing
    images = preprocess_input(images)


    print(
        "\nValidation Dataset "
        "Loaded Successfully"
    )


    print(
        "Image Shape:",
        images.shape
    )


    print(
        "Label Shape:",
        labels.shape
    )


    return (
        images,
        labels
    )


# ============================================================
# LOAD TRAINED VGG16 MODEL
# ============================================================

def load_model():

    print(
        "\nLoading trained "
        "VGG16 model..."
    )


    if not os.path.exists(
            MODEL_PATH
    ):

        raise FileNotFoundError(

            f"VGG16 model not found:\n"
            f"{MODEL_PATH}"

        )


    model = (
        tf.keras.models.load_model(
            MODEL_PATH
        )
    )


    print(
        "VGG16 Model "
        "Loaded Successfully"
    )


    return model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
        model,
        validation_images,
        validation_labels
):

    print(
        "\n" + "=" * 70
    )


    print(
        "EVALUATING VGG16 MODEL"
    )


    print(
        "=" * 70
    )


    loss, accuracy = model.evaluate(

        validation_images,

        validation_labels,

        verbose=1

    )


    print(
        "\nFinal Validation Loss:",
        loss
    )


    print(
        "Final Validation Accuracy:",
        accuracy
    )


    return (
        loss,
        accuracy
    )


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
        model,
        validation_images
):

    print(
        "\nGenerating predictions..."
    )


    predictions = model.predict(

        validation_images,

        verbose=1

    )


    predicted_labels = np.argmax(

        predictions,

        axis=1

    )


    print(
        "Predictions Generated "
        "Successfully"
    )


    return (
        predicted_labels,
        predictions
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def create_classification_report(
        true_labels,
        predicted_labels,
        classes
):

    print(
        "\nGenerating "
        "Classification Report..."
    )


    report = classification_report(

        true_labels,

        predicted_labels,

        labels=list(
            range(
                len(classes)
            )
        ),

        target_names=classes,

        digits=4,

        zero_division=0

    )


    report_path = os.path.join(

        METRICS_DIR,

        "vgg16_classification_report.txt"

    )


    with open(

            report_path,

            "w",

            encoding="utf-8"

    ) as file:

        file.write(
            report
        )


    print(
        "\nClassification Report:"
    )


    print(
        report
    )


    print(
        f"\nClassification "
        f"Report Saved:\n"
        f"{report_path}"
    )


    return report


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
        true_labels,
        predicted_labels,
        classes
):

    print(
        "\nCreating Confusion Matrix..."
    )


    matrix = confusion_matrix(

        true_labels,

        predicted_labels,

        labels=list(
            range(
                len(classes)
            )
        )

    )


    plt.figure(
        figsize=(
            20,
            18
        )
    )


    plt.imshow(
        matrix
    )


    plt.title(
        "VGG16 Food Classification "
        "Confusion Matrix"
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

        "vgg16_confusion_matrix.png"

    )


    plt.savefig(

        matrix_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    print(
        f"Confusion Matrix Saved:\n"
        f"{matrix_path}"
    )


    return matrix


# ============================================================
# PER CLASS ACCURACY
# ============================================================

def calculate_per_class_accuracy(
        matrix,
        classes
):

    print(
        "\n" + "=" * 70
    )


    print(
        "PER CLASS ACCURACY"
    )


    print(
        "=" * 70
    )


    class_results = []


    for index, class_name in enumerate(
            classes
    ):


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

            "class_name":
                class_name,

            "total_validation_images":
                int(
                    total_images
                ),

            "correct_predictions":
                int(
                    correct_predictions
                ),

            "accuracy_percent":
                round(
                    float(
                        accuracy
                    ),
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


    # Sort from lowest accuracy
    # to highest accuracy

    sorted_results = sorted(

        class_results,

        key=lambda item:
            item[
                "accuracy_percent"
            ]

    )


    results_path = os.path.join(

        METRICS_DIR,

        "vgg16_per_class_accuracy.json"

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


    print(
        "\nPer Class Accuracy Saved:"
    )


    print(
        results_path
    )


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

        "model_name":
            "VGG16",

        "total_classes":
            34,

        "validation_loss":
            round(
                float(
                    validation_loss
                ),
                6
            ),

        "validation_accuracy_percent":
            round(
                float(
                    overall_accuracy
                ),
                2
            ),

        "weakest_classes":
            weakest_classes,

        "strongest_classes":
            strongest_classes

    }


    summary_path = os.path.join(

        METRICS_DIR,

        "vgg16_evaluation_summary.json"

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


    print(
        "\nEvaluation Summary Saved:"
    )


    print(
        summary_path
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )


    print(
        "FOOD CLASSIFICATION VGG16 "
        "- MODEL EVALUATION"
    )


    print(
        "=" * 70
    )


    # Load classes

    classes = load_classes()


    # Load validation data

    validation_images, validation_labels = (

        load_validation_data()

    )


    # Load trained VGG16 model

    model = load_model()


    # Evaluate model

    validation_loss, validation_accuracy = (

        evaluate_model(

            model,

            validation_images,

            validation_labels

        )

    )


    # Generate predictions

    predicted_labels, predictions = (

        generate_predictions(

            model,

            validation_images

        )

    )


    # Classification report

    create_classification_report(

        validation_labels,

        predicted_labels,

        classes

    )


    # Confusion matrix

    matrix = create_confusion_matrix(

        validation_labels,

        predicted_labels,

        classes

    )


    # Per class accuracy

    per_class_results = (

        calculate_per_class_accuracy(

            matrix,

            classes

        )

    )


    # Save evaluation summary

    save_evaluation_summary(

        validation_loss,

        validation_accuracy,

        per_class_results

    )


    print(
        "\n" + "=" * 70
    )


    print(
        "VGG16 MODEL EVALUATION "
        "COMPLETED SUCCESSFULLY!"
    )


    print(
        "=" * 70
    )


    print(

        f"\nVGG16 Model Accuracy: "

        f"{validation_accuracy * 100:.2f}%"

    )


    print(
        "\nAll evaluation files "
        "are available in:"
    )


    print(
        METRICS_DIR
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()