import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from tensorflow.keras.applications.resnet50 import (
    preprocess_input
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


os.makedirs(
    METRICS_DIR,
    exist_ok=True
)


# ============================================================
# FILE PATHS
# ============================================================

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
    "best_food_resnet50.keras"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (
    224,
    224
)

BATCH_SIZE = 16


# ============================================================
# LOAD CLASS INFORMATION
# ============================================================

def load_classes():

    print("\nLoading class information...")

    if not os.path.exists(
            CLASSES_JSON
    ):
        raise FileNotFoundError(
            f"Classes JSON not found:\n"
            f"{CLASSES_JSON}"
        )

    with open(
            CLASSES_JSON,
            "r",
            encoding="utf-8"
    ) as file:

        class_data = json.load(
            file
        )


    print(
        f"\nClasses JSON Type: "
        f"{type(class_data)}"
    )


    # --------------------------------------------------------
    # Extract "classes" field
    # --------------------------------------------------------

    if (
        isinstance(
            class_data,
            dict
        )
        and
        "classes" in class_data
    ):

        classes_data = class_data[
            "classes"
        ]

    else:

        classes_data = class_data


    # --------------------------------------------------------
    # Case 1: List
    # --------------------------------------------------------

    if isinstance(
            classes_data,
            list
    ):

        # If list contains dictionaries
        if (
            len(classes_data) > 0
            and
            isinstance(
                classes_data[0],
                dict
            )
        ):

            sorted_classes = sorted(
                classes_data,
                key=lambda item:
                    item[
                        "class_index"
                    ]
            )

            classes = [

                item[
                    "class_name"
                ]

                for item in sorted_classes

            ]

        else:

            classes = classes_data


    # --------------------------------------------------------
    # Case 2: Dictionary
    # --------------------------------------------------------

    elif isinstance(
            classes_data,
            dict
    ):

        # Example:
        # {
        #     "apple_pie": 0,
        #     "burger": 1
        # }

        if all(
                isinstance(
                    value,
                    int
                )
                for value
                in classes_data.values()
        ):

            sorted_items = sorted(

                classes_data.items(),

                key=lambda item:
                    item[1]

            )

            classes = [

                item[0]

                for item
                in sorted_items

            ]


        # Example:
        # {
        #     "0": "apple_pie",
        #     "1": "burger"
        # }

        else:

            sorted_items = sorted(

                classes_data.items(),

                key=lambda item:
                    int(
                        item[0]
                    )

            )

            classes = [

                item[1]

                for item
                in sorted_items

            ]


    else:

        raise ValueError(
            "Unsupported classes.json structure"
        )


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
            f"{index}: "
            f"{class_name}"
        )


    return classes


# ============================================================
# LOAD VALIDATION JSON RECORDS
# ============================================================

def load_validation_records():

    print(
        "\nLoading validation dataset records..."
    )


    if not os.path.exists(
            VALIDATION_JSON
    ):

        raise FileNotFoundError(

            f"Validation JSON not found:\n"
            f"{VALIDATION_JSON}"

        )


    with open(

            VALIDATION_JSON,

            "r",

            encoding="utf-8"

    ) as file:

        records = json.load(
            file
        )


    print(
        f"Total Validation Records: "
        f"{len(records)}"
    )


    return records


# ============================================================
# LOAD AND PREPROCESS IMAGE
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


    # --------------------------------------------------------
    # IMPORTANT
    #
    # Same preprocessing used during ResNet50 training
    # --------------------------------------------------------

    image = preprocess_input(
        image
    )


    return image, label


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

def create_validation_dataset(
        records
):

    print(
        "\nCreating validation dataset..."
    )


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


        if not os.path.exists(
                full_path
        ):

            raise FileNotFoundError(

                f"Validation image not found:\n"
                f"{full_path}"

            )


        image_paths.append(
            full_path
        )


        labels.append(

            record[
                "class_index"
            ]

        )


    labels = np.array(

        labels,

        dtype=np.int32

    )


    dataset = tf.data.Dataset.from_tensor_slices(

        (

            image_paths,

            labels

        )

    )


    dataset = dataset.map(

        load_and_preprocess_image,

        num_parallel_calls=
            tf.data.AUTOTUNE

    )


    dataset = dataset.batch(
        BATCH_SIZE
    )


    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )


    print(
        "Validation Dataset Ready"
    )


    print(
        f"Total Validation Images: "
        f"{len(labels)}"
    )


    return dataset, labels


# ============================================================
# LOAD TRAINED RESNET50 MODEL
# ============================================================

def load_model():

    print(
        "\nLoading trained ResNet50 model..."
    )


    if not os.path.exists(
            MODEL_PATH
    ):

        raise FileNotFoundError(

            f"ResNet50 model not found:\n"
            f"{MODEL_PATH}"

        )


    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    print(
        "ResNet50 Model Loaded Successfully"
    )


    print(
        "\nModel Input Shape:"
    )

    print(
        model.input_shape
    )


    print(
        "\nModel Output Shape:"
    )

    print(
        model.output_shape
    )


    return model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
        model,
        validation_dataset
):

    print(
        "\n" + "=" * 70
    )

    print(
        "EVALUATING RESNET50 MODEL"
    )

    print(
        "=" * 70
    )


    loss, accuracy = model.evaluate(

        validation_dataset,

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


    return loss, accuracy


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
        model,
        validation_dataset
):

    print(
        "\nGenerating predictions..."
    )


    predictions = model.predict(

        validation_dataset,

        verbose=1

    )


    predicted_labels = np.argmax(

        predictions,

        axis=1

    )


    print(
        "Predictions Generated Successfully"
    )


    return (
        predicted_labels,
        predictions
    )


# ============================================================
# CREATE CLASSIFICATION REPORT
# ============================================================

def create_classification_report(

        true_labels,

        predicted_labels,

        classes

):

    print(
        "\nGenerating Classification Report..."
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

        "resnet50_classification_report.txt"

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
        "\nClassification Report Saved:"
    )


    print(
        report_path
    )


    return report


# ============================================================
# CREATE CONFUSION MATRIX
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
        "ResNet50 Food Classification Confusion Matrix"
    )


    plt.colorbar()


    tick_positions = np.arange(
        len(classes)
    )


    plt.xticks(

        tick_positions,

        classes,

        rotation=90,

        fontsize=8

    )


    plt.yticks(

        tick_positions,

        classes,

        fontsize=8

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

        "resnet50_confusion_matrix.png"

    )


    plt.savefig(

        matrix_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    print(
        "Confusion Matrix Saved:"
    )


    print(
        matrix_path
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
        "RESNET50 PER CLASS ACCURACY"
    )

    print(
        "=" * 70
    )


    class_results = []


    for index, class_name in enumerate(
            classes
    ):


        total_images = np.sum(
            matrix[
                index
            ]
        )


        correct_predictions = matrix[
            index,
            index
        ]


        if total_images > 0:


            accuracy = (

                correct_predictions
                /
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


    # --------------------------------------------------------
    # Sort Lowest Accuracy to Highest Accuracy
    # --------------------------------------------------------

    sorted_results = sorted(

        class_results,

        key=lambda x:
            x[
                "accuracy_percent"
            ]

    )


    results_path = os.path.join(

        METRICS_DIR,

        "resnet50_per_class_accuracy.json"

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

        per_class_results,

        total_classes

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


        "model":
            "ResNet50",


        "total_classes":
            int(
                total_classes
            ),


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

        "resnet50_evaluation_summary.json"

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
        "FOOD CLASSIFICATION RESNET50 - MODEL EVALUATION"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Load Classes
    # --------------------------------------------------------

    classes = load_classes()


    # --------------------------------------------------------
    # Load Validation Records
    # --------------------------------------------------------

    validation_records = (
        load_validation_records()
    )


    # --------------------------------------------------------
    # Create Validation Dataset
    # --------------------------------------------------------

    validation_dataset, validation_labels = (

        create_validation_dataset(
            validation_records
        )

    )


    # --------------------------------------------------------
    # Load Model
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # Evaluate Model
    # --------------------------------------------------------

    validation_loss, validation_accuracy = (

        evaluate_model(

            model,

            validation_dataset

        )

    )


    # --------------------------------------------------------
    # Generate Predictions
    # --------------------------------------------------------

    predicted_labels, predictions = (

        generate_predictions(

            model,

            validation_dataset

        )

    )


    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    create_classification_report(

        validation_labels,

        predicted_labels,

        classes

    )


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    matrix = create_confusion_matrix(

        validation_labels,

        predicted_labels,

        classes

    )


    # --------------------------------------------------------
    # Per Class Accuracy
    # --------------------------------------------------------

    per_class_results = (

        calculate_per_class_accuracy(

            matrix,

            classes

        )

    )


    # --------------------------------------------------------
    # Evaluation Summary
    # --------------------------------------------------------

    save_evaluation_summary(

        validation_loss,

        validation_accuracy,

        per_class_results,

        len(
            classes
        )

    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RESNET50 MODEL EVALUATION COMPLETED SUCCESSFULLY!"
    )

    print(
        "=" * 70
    )


    print(

        f"\nResNet50 Model Accuracy: "

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