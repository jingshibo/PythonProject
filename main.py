import os

# Set the root directories where duplicate files exist
root_folders = ["E:\OneDrive\文档"]

def delete_copies(folder_path):
    """
    Recursively walks through all subdirectories and deletes files with '- Copy' in their names.
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if "- Copy" in file:
                file_path = os.path.join(root, file)
                print(f"Deleting: {file_path}")
                os.remove(file_path)  # Delete the file

# Process both folders
for folder in root_folders:
    delete_copies(folder)

print("All duplicate '- Copy' files have been deleted!")
