import os
import json

ROOT_DIR = ".."   # project root
OUTPUT_FILE = "project_structure.json"

def build_tree(path):
    tree = {}

    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        return "PermissionDenied"

    for item in items:
        full_path = os.path.join(path, item)

        if os.path.isdir(full_path):
            tree[item] = build_tree(full_path)
        else:
            tree[item] = "file"

    return tree


project_tree = build_tree(ROOT_DIR)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(project_tree, f, indent=4)

print(f"\nProject structure saved to {OUTPUT_FILE}")