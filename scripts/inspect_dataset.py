from pathlib import Path
from collections import Counter
from PIL import Image


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "Food Classification dataset"
)

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ==========================================
# GET CLASS FOLDERS
# ==========================================

class_folders = sorted(
    [
        folder
        for folder in DATASET_PATH.iterdir()
        if folder.is_dir()
    ]
)


print("\n" + "=" * 70)
print("FOOD DATASET INSPECTION REPORT")
print("=" * 70)

print(f"\nDataset Path: {DATASET_PATH}")
print(f"Total Classes Found: {len(class_folders)}")


# ==========================================
# INSPECT DATASET
# ==========================================

total_images = 0
extension_counter = Counter()
corrupted_images = []
class_image_counts = {}


for class_folder in class_folders:

    class_name = class_folder.name

    image_files = [
        file
        for file in class_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    ]

    image_count = len(image_files)

    class_image_counts[class_name] = image_count

    total_images += image_count

    for image_file in image_files:

        extension_counter[
            image_file.suffix.lower()
        ] += 1

        try:
            with Image.open(image_file) as img:
                img.verify()

        except Exception:
            corrupted_images.append(str(image_file))


# ==========================================
# PRINT CLASS REPORT
# ==========================================

print("\n" + "-" * 70)
print("IMAGES PER CLASS")
print("-" * 70)

for class_name, count in class_image_counts.items():

    status = "OK" if count >= 300 else "LESS THAN 300"

    print(
        f"{class_name:<25} "
        f"{count:>5} images   "
        f"[{status}]"
    )


# ==========================================
# FINAL REPORT
# ==========================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(f"\nTotal Classes       : {len(class_folders)}")
print(f"Total Images        : {total_images}")

print("\nImage Extensions:")

for extension, count in extension_counter.items():
    print(f"  {extension}: {count}")


print(f"\nCorrupted Images: {len(corrupted_images)}")


# ==========================================
# CHECK 300 IMAGES REQUIREMENT
# ==========================================

less_than_300 = {
    class_name: count
    for class_name, count in class_image_counts.items()
    if count < 300
}


print("\n" + "=" * 70)
print("300 IMAGES REQUIREMENT CHECK")
print("=" * 70)


if less_than_300:

    print("\nWARNING: Classes with less than 300 images:\n")

    for class_name, count in less_than_300.items():
        print(f"{class_name:<25} {count} images")

else:

    print("\nSUCCESS! All classes have at least 300 images.")


print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)