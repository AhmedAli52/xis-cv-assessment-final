import os
from pathlib import Path

# Get the folder where THIS script is located
SCRIPT_DIR = Path(__file__).resolve().parent

# If raw_images is inside the same folder as this script
IMAGE_DIR = SCRIPT_DIR / "raw_images"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def main():
    if not IMAGE_DIR.exists():
        print(f"Folder not found: {IMAGE_DIR}")
        return

    image_files = [
        f for f in IMAGE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]

    if not image_files:
        print("No image files found.")
        return

    image_files.sort()

    print(f"Found {len(image_files)} images.\n")

    for index, old_file in enumerate(image_files, start=1):
        new_name = f"book_{index:03d}{old_file.suffix.lower()}"
        new_file = IMAGE_DIR / new_name

        old_file.rename(new_file)
        print(f"{old_file.name}  -->  {new_name}")

    print("\nRenaming completed successfully.")

if __name__ == "__main__":
    main()