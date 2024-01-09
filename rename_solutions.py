import os

def rename_files(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file == "solutions_answer_key.py":
                old_file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, "solutions.py")
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {old_file_path} to {new_file_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py [path_to_root_folder]")
        sys.exit(1)

    root_folder_path = sys.argv[1]
    rename_files(root_folder_path)
