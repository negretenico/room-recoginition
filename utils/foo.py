import os
import shutil


def organize_files(input_directory):
    # List all files in the input directory
    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

    # Create subdirectories based on prefixes
    for filename in files:
        prefix, extension = os.path.splitext(filename)
        prefix = prefix.split("_")[0]
        target_directory = os.path.join(input_directory, prefix)

        # Create the target directory if it doesn't exist
        os.makedirs(target_directory, exist_ok=True)

        # Move the file to the target directory
        source_path = os.path.join(input_directory, filename)
        target_path = os.path.join(target_directory, filename)
        shutil.move(source_path, target_path)


if __name__ == "__main__":
    input_directory = os.path.join('multiclass')
    organize_files(input_directory)
