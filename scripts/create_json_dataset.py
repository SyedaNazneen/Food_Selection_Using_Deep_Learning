import json
from pathlib import Path
from datetime import datetime


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BALANCED_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "balanced_dataset"
)

JSON_OUTPUT_FOLDER = PROJECT_ROOT / "json_data"

TRAIN_FOLDER = BALANCED_DATASET / "train"

VALIDATION_FOLDER = BALANCED_DATASET / "validation"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

JSON_OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# GET IMAGE FILES
# ==========================================================

def get_image_files(folder):

    images = []

    for file in sorted(folder.iterdir()):

        if (
            file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        ):
            images.append(file)

    return images


# ==========================================================
# CREATE DATASET RECORDS
# ==========================================================

def create_records(dataset_folder, dataset_type, class_to_index):

    records = []

    class_folders = sorted(
        [
            folder
            for folder in dataset_folder.iterdir()
            if folder.is_dir()
        ],
        key=lambda x: x.name.lower()
    )

    for class_folder in class_folders:

        class_name = class_folder.name

        class_index = class_to_index[class_name]

        images = get_image_files(class_folder)

        for image_number, image_path in enumerate(
            images,
            start=1
        ):

            relative_path = image_path.relative_to(
                PROJECT_ROOT
            ).as_posix()

            record = {

                "image_id": (
                    f"{dataset_type}_"
                    f"{class_index}_"
                    f"{image_number:03d}"
                ),

                "class_name": class_name,

                "class_index": class_index,

                "dataset_type": dataset_type,

                "image_path": relative_path

            }

            records.append(record)

    return records


# ==========================================================
# SAVE JSON FILE
# ==========================================================

def save_json(data, output_file):

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def main():

    print("\n" + "=" * 70)

    print(
        "FOOD DATASET JSON CREATION"
    )

    print("=" * 70)

    # ------------------------------------------------------
    # CHECK DATASET
    # ------------------------------------------------------

    if not TRAIN_FOLDER.exists():

        print(
            "\nERROR: Training folder not found!"
        )

        print(TRAIN_FOLDER)

        return

    if not VALIDATION_FOLDER.exists():

        print(
            "\nERROR: Validation folder not found!"
        )

        print(VALIDATION_FOLDER)

        return

    # ------------------------------------------------------
    # GET CLASSES
    # ------------------------------------------------------

    class_folders = sorted(
        [
            folder
            for folder in TRAIN_FOLDER.iterdir()
            if folder.is_dir()
        ],
        key=lambda x: x.name.lower()
    )

    class_names = [
        folder.name
        for folder in class_folders
    ]

    class_to_index = {

        class_name: index

        for index, class_name in enumerate(
            class_names
        )
    }

    # ------------------------------------------------------
    # CREATE TRAIN RECORDS
    # ------------------------------------------------------

    print(
        "\nCreating Training JSON Records..."
    )

    train_records = create_records(

        TRAIN_FOLDER,

        "train",

        class_to_index
    )

    # ------------------------------------------------------
    # CREATE VALIDATION RECORDS
    # ------------------------------------------------------

    print(
        "Creating Validation JSON Records..."
    )

    validation_records = create_records(

        VALIDATION_FOLDER,

        "validation",

        class_to_index
    )

    # ------------------------------------------------------
    # CREATE CLASSES DATA
    # ------------------------------------------------------

    classes_data = {

        "total_classes": len(
            class_names
        ),

        "classes": [

            {

                "class_name": class_name,

                "class_index": class_to_index[
                    class_name
                ]

            }

            for class_name in class_names

        ]

    }

    # ------------------------------------------------------
    # DATASET SUMMARY
    # ------------------------------------------------------

    dataset_summary = {

        "project_name": (
            "Food Selection Using "
            "Deep Learning CNN"
        ),

        "created_at": (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ),

        "total_classes": len(
            class_names
        ),

        "training_images": len(
            train_records
        ),

        "validation_images": len(
            validation_records
        ),

        "total_images": (

            len(train_records)

            +

            len(validation_records)

        ),

        "images_per_class": {

            "training": 250,

            "validation": 50

        }

    }

    # ------------------------------------------------------
    # SAVE JSON FILES
    # ------------------------------------------------------

    print(
        "\nSaving JSON Files..."
    )

    save_json(

        train_records,

        JSON_OUTPUT_FOLDER
        / "train_dataset.json"

    )

    save_json(

        validation_records,

        JSON_OUTPUT_FOLDER
        / "validation_dataset.json"

    )

    save_json(

        classes_data,

        JSON_OUTPUT_FOLDER
        / "classes.json"

    )

    save_json(

        dataset_summary,

        JSON_OUTPUT_FOLDER
        / "dataset_summary.json"

    )

    # ------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "JSON CREATION SUMMARY"
    )

    print("=" * 70)

    print(
        f"\nTotal Classes: "
        f"{len(class_names)}"
    )

    print(
        f"Training Records: "
        f"{len(train_records)}"
    )

    print(
        f"Validation Records: "
        f"{len(validation_records)}"
    )

    print(
        f"Total Records: "
        f"{len(train_records) + len(validation_records)}"
    )

    print(
        "\nJSON Files Created:"
    )

    print(
        "1. train_dataset.json"
    )

    print(
        "2. validation_dataset.json"
    )

    print(
        "3. classes.json"
    )

    print(
        "4. dataset_summary.json"
    )

    print(
        "\nOutput Folder:"
    )

    print(
        JSON_OUTPUT_FOLDER
    )

    print("\n" + "=" * 70)

    print(
        "SUCCESS! JSON FILES CREATED."
    )

    print("=" * 70)


# ==========================================================
# RUN PROGRAM
# ==========================================================

if __name__ == "__main__":

    main()