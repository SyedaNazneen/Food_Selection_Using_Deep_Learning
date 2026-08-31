from pathlib import Path
import random
import shutil
from collections import Counter

from PIL import Image, ImageOps, ImageEnhance


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "Food Classification dataset"
)

OUTPUT_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "balanced_dataset"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TARGET_IMAGES_PER_CLASS = 300

TRAIN_IMAGES_PER_CLASS = 250
VALIDATION_IMAGES_PER_CLASS = 50

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

random.seed(RANDOM_SEED)


# ============================================================
# IMAGE AUGMENTATION FUNCTION
# ============================================================

def augment_image(image_path, output_path, augmentation_number):

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        operation = augmentation_number % 6

        # 0 -> Horizontal Flip
        if operation == 0:
            augmented = ImageOps.mirror(image)

        # 1 -> Slight Rotation
        elif operation == 1:
            augmented = image.rotate(
                10,
                resample=Image.Resampling.BICUBIC
            )

        # 2 -> Slight Rotation Opposite Direction
        elif operation == 2:
            augmented = image.rotate(
                -10,
                resample=Image.Resampling.BICUBIC
            )

        # 3 -> Brightness Change
        elif operation == 3:
            enhancer = ImageEnhance.Brightness(image)
            augmented = enhancer.enhance(1.2)

        # 4 -> Contrast Change
        elif operation == 4:
            enhancer = ImageEnhance.Contrast(image)
            augmented = enhancer.enhance(1.2)

        # 5 -> Slight Brightness Reduction
        else:
            enhancer = ImageEnhance.Brightness(image)
            augmented = enhancer.enhance(0.8)

        augmented.save(
            output_path,
            format="JPEG",
            quality=95
        )


# ============================================================
# CLEAN OLD OUTPUT DATASET
# ============================================================

if OUTPUT_DATASET.exists():

    print("\nOld balanced_dataset found.")
    print("Deleting old balanced_dataset...")

    shutil.rmtree(OUTPUT_DATASET)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

TRAIN_PATH = OUTPUT_DATASET / "train"

VALIDATION_PATH = OUTPUT_DATASET / "validation"

TRAIN_PATH.mkdir(
    parents=True,
    exist_ok=True
)

VALIDATION_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GET ALL CLASS FOLDERS
# ============================================================

class_folders = sorted(
    [
        folder
        for folder in SOURCE_DATASET.iterdir()
        if folder.is_dir()
    ]
)


print("\n" + "=" * 70)
print("BALANCED DATASET PREPARATION")
print("=" * 70)

print(f"\nTotal Classes Found: {len(class_folders)}")

print(
    f"Target per Class: "
    f"{TARGET_IMAGES_PER_CLASS}"
)

print(
    f"Training per Class: "
    f"{TRAIN_IMAGES_PER_CLASS}"
)

print(
    f"Validation per Class: "
    f"{VALIDATION_IMAGES_PER_CLASS}"
)


# ============================================================
# REPORT VARIABLES
# ============================================================

final_report = {}

total_train_images = 0
total_validation_images = 0
total_augmented_images = 0


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for class_folder in class_folders:

    class_name = class_folder.name

    print("\n" + "-" * 70)
    print(f"Processing Class: {class_name}")
    print("-" * 70)


    # --------------------------------------------------------
    # GET VALID IMAGE FILES
    # --------------------------------------------------------

    image_files = sorted(
        [
            file
            for file in class_folder.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in VALID_EXTENSIONS
        ]
    )


    original_count = len(image_files)

    print(
        f"Original Images Available: "
        f"{original_count}"
    )


    # --------------------------------------------------------
    # SHUFFLE IMAGES
    # --------------------------------------------------------

    random.shuffle(image_files)


    # --------------------------------------------------------
    # SELECT VALIDATION IMAGES
    # --------------------------------------------------------

    validation_files = image_files[
        :VALIDATION_IMAGES_PER_CLASS
    ]

    remaining_files = image_files[
        VALIDATION_IMAGES_PER_CLASS:
    ]


    # --------------------------------------------------------
    # CREATE CLASS OUTPUT FOLDERS
    # --------------------------------------------------------

    train_class_path = (
        TRAIN_PATH
        / class_name
    )

    validation_class_path = (
        VALIDATION_PATH
        / class_name
    )

    train_class_path.mkdir(
        exist_ok=True
    )

    validation_class_path.mkdir(
        exist_ok=True
    )


    # --------------------------------------------------------
    # COPY VALIDATION IMAGES
    # --------------------------------------------------------

    for index, image_file in enumerate(
        validation_files,
        start=1
    ):

        destination = (
            validation_class_path
            / f"val_{index:03d}.jpg"
        )

        with Image.open(image_file) as image:

            image = image.convert("RGB")

            image.save(
                destination,
                format="JPEG",
                quality=95
            )


    # --------------------------------------------------------
    # SELECT ORIGINAL TRAINING IMAGES
    # --------------------------------------------------------

    available_for_training = min(
        len(remaining_files),
        TRAIN_IMAGES_PER_CLASS
    )

    training_original_files = (
        remaining_files[
            :available_for_training
        ]
    )


    # --------------------------------------------------------
    # COPY ORIGINAL TRAINING IMAGES
    # --------------------------------------------------------

    for index, image_file in enumerate(
        training_original_files,
        start=1
    ):

        destination = (
            train_class_path
            / f"train_original_{index:03d}.jpg"
        )

        with Image.open(image_file) as image:

            image = image.convert("RGB")

            image.save(
                destination,
                format="JPEG",
                quality=95
            )


    # --------------------------------------------------------
    # AUGMENT TRAINING IMAGES IF NEEDED
    # --------------------------------------------------------

    augmented_needed = (
        TRAIN_IMAGES_PER_CLASS
        - len(training_original_files)
    )

    print(
        f"Original Training Images: "
        f"{len(training_original_files)}"
    )

    print(
        f"Augmented Images Needed: "
        f"{augmented_needed}"
    )


    # If augmentation is required
    if augmented_needed > 0:

        if len(training_original_files) == 0:

            raise ValueError(
                f"Not enough images available "
                f"for class: {class_name}"
            )


        augmentation_sources = (
            training_original_files.copy()
        )

        augmentation_index = 1


        while augmented_needed > 0:

            source_image = random.choice(
                augmentation_sources
            )

            destination = (
                train_class_path
                / (
                    f"train_augmented_"
                    f"{augmentation_index:03d}.jpg"
                )
            )

            augment_image(
                source_image,
                destination,
                augmentation_index
            )

            augmented_needed -= 1

            augmentation_index += 1


    # --------------------------------------------------------
    # FINAL COUNT
    # --------------------------------------------------------

    train_count = len(
        list(
            train_class_path.glob(
                "*.jpg"
            )
        )
    )

    validation_count = len(
        list(
            validation_class_path.glob(
                "*.jpg"
            )
        )
    )

    augmented_count = (
        train_count
        - len(training_original_files)
    )


    total_train_images += train_count

    total_validation_images += validation_count

    total_augmented_images += augmented_count


    final_report[class_name] = {

        "original_available":
            original_count,

        "original_training":
            len(
                training_original_files
            ),

        "augmented_training":
            augmented_count,

        "final_training":
            train_count,

        "validation":
            validation_count
    }


    print(
        f"Final Training Images: "
        f"{train_count}"
    )

    print(
        f"Final Validation Images: "
        f"{validation_count}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL BALANCED DATASET SUMMARY")
print("=" * 70)


print(
    f"\nTotal Classes: "
    f"{len(class_folders)}"
)

print(
    f"Total Training Images: "
    f"{total_train_images}"
)

print(
    f"Total Validation Images: "
    f"{total_validation_images}"
)

print(
    f"Total Augmented Images: "
    f"{total_augmented_images}"
)

print(
    f"\nExpected Training Images: "
    f"{len(class_folders) * TRAIN_IMAGES_PER_CLASS}"
)

print(
    f"Expected Validation Images: "
    f"{len(class_folders) * VALIDATION_IMAGES_PER_CLASS}"
)

print(
    f"Expected Total Images: "
    f"{len(class_folders) * TARGET_IMAGES_PER_CLASS}"
)


# ============================================================
# VERIFY REQUIREMENTS
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)


verification_success = True


for class_name, details in final_report.items():

    train_ok = (
        details["final_training"]
        == TRAIN_IMAGES_PER_CLASS
    )

    validation_ok = (
        details["validation"]
        == VALIDATION_IMAGES_PER_CLASS
    )


    if train_ok and validation_ok:

        print(
            f"{class_name:<25} "
            f"TRAIN: {details['final_training']:3d} "
            f"VALIDATION: {details['validation']:3d} "
            f"[OK]"
        )

    else:

        verification_success = False

        print(
            f"{class_name:<25} "
            f"[ERROR]"
        )


print("\n" + "=" * 70)

if verification_success:

    print(
        "SUCCESS! BALANCED DATASET "
        "CREATED CORRECTLY."
    )

else:

    print(
        "ERROR! SOME CLASSES DO NOT "
        "MATCH THE REQUIRED COUNTS."
    )

print("=" * 70)

print(
    "\nOutput Dataset Location:"
)

print(
    OUTPUT_DATASET
)