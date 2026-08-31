import os
import json
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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
# MODEL RESULTS
# ============================================================

model_results = {

    "Custom CNN": {
        "accuracy": 64.35
    },

    "VGG16": {
        "accuracy": 75.12
    },

    "ResNet50": {
        "accuracy": 82.94
    }

}


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "FOOD CLASSIFICATION MODEL COMPARISON"
    )

    print(
        "CNN vs VGG16 vs ResNet50"
    )

    print("=" * 70)


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\nMODEL PERFORMANCE COMPARISON\n")

    for model_name, results in model_results.items():

        print(
            f"{model_name:<15} : "
            f"{results['accuracy']:.2f}%"
        )


    # ========================================================
    # FIND BEST MODEL
    # ========================================================

    best_model = max(

        model_results,

        key=lambda model:
        model_results[model]["accuracy"]

    )

    best_accuracy = model_results[
        best_model
    ]["accuracy"]


    print("\n" + "=" * 70)

    print(
        f"BEST MODEL: {best_model}"
    )

    print(
        f"BEST ACCURACY: {best_accuracy:.2f}%"
    )

    print("=" * 70)


    # ========================================================
    # SAVE COMPARISON JSON
    # ========================================================

    comparison_data = {

        "models": model_results,

        "best_model": best_model,

        "best_accuracy": best_accuracy

    }


    comparison_json_path = os.path.join(

        METRICS_DIR,

        "model_comparison_summary.json"

    )


    with open(

        comparison_json_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            comparison_data,

            file,

            indent=4

        )


    print(
        "\nComparison Summary Saved:"
    )

    print(
        comparison_json_path
    )


    # ========================================================
    # ACCURACY COMPARISON GRAPH
    # ========================================================

    model_names = list(
        model_results.keys()
    )

    accuracies = [

        model_results[
            model
        ]["accuracy"]

        for model in model_names

    ]


    plt.figure(
        figsize=(10, 6)
    )


    bars = plt.bar(

        model_names,

        accuracies

    )


    plt.title(

        "Food Classification Model Accuracy Comparison"

    )


    plt.xlabel(

        "Models"

    )


    plt.ylabel(

        "Validation Accuracy (%)"

    )


    plt.ylim(
        0,
        100
    )


    plt.grid(
        axis="y"
    )


    # Add accuracy labels

    for bar, accuracy in zip(
            bars,
            accuracies
    ):

        plt.text(

            bar.get_x()
            +
            bar.get_width() / 2,

            bar.get_height() + 1,

            f"{accuracy:.2f}%",

            ha="center",

            fontsize=11,

            fontweight="bold"

        )


    graph_path = os.path.join(

        METRICS_DIR,

        "cnn_vgg16_resnet50_comparison.png"

    )


    plt.savefig(

        graph_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    print(
        "\nComparison Graph Saved:"
    )

    print(
        graph_path
    )


    # ========================================================
    # FINAL CONCLUSION
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "FINAL PROJECT CONCLUSION"
    )

    print("=" * 70)


    print(

        f"""

1. Custom CNN Accuracy : 64.35%

2. VGG16 Accuracy      : 75.12%

3. ResNet50 Accuracy   : 82.94%

Best Performing Model  : {best_model}

Conclusion:
ResNet50 achieved the highest validation accuracy
and performed better than both Custom CNN and VGG16
for the 34-class Food Classification Dataset.

"""

    )


    print("=" * 70)

    print(
        "MODEL COMPARISON COMPLETED SUCCESSFULLY!"
    )

    print("=" * 70)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()