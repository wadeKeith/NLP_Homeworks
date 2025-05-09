import os
import shutil

root_dir = "./"

for dirpath, dirnames, filenames in os.walk(root_dir):
    if "__pycache__" in dirnames:
        pycache_dir = os.path.join(dirpath, "__pycache__")

        shutil.rmtree(pycache_dir)

        print(f"Removed: {pycache_dir}")

    if ".DS_Store" in filenames:
        ds_store_file = os.path.join(dirpath, ".DS_Store")

        os.remove(ds_store_file)

        print(f"Removed: {ds_store_file}")



