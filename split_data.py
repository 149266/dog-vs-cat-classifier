
import os
import shutil
import random

# --- Original mappe ---
base_dir = "dog_vs_cat_data"
source_cat = "/Users/ingelin/dog_vs_cat_data/PetImages/cat"
source_dog = "/Users/ingelin/dog_vs_cat_data/PetImages/dog"

# --- Målmapper ---
split_dir = "dog_vs_cat_data_split2"
groups = ["train_group1.1", "train_group2.2", "validation.2"]
classes = ["cat", "dog"]    

# --- Lag mapper ---
for group in groups:
    for cls in classes:
        os.makedirs(os.path.join(split_dir, group, cls), exist_ok=True)

# --- Funksjon for å splitte bilder ---
def split_class_images(source_dir, cls):
    images = os.listdir(source_dir)
    random.shuffle(images)

    n = len(images)
    n_train1 = int(n * 0.4)  # 40% til train_group1
    n_train2 = int(n * 0.4)  # 40% til train_group2
    n_val = n - n_train1 - n_train2  # resten til validation

    for i, img in enumerate(images):
        src = os.path.join(source_dir, img)
        if i < n_train1:
            dst = os.path.join(split_dir, "train_group1.1", cls, img)
        elif i < n_train1 + n_train2:
            dst = os.path.join(split_dir, "train_group2.2", cls, img)
        else:
            dst = os.path.join(split_dir, "validation.2", cls, img)
        shutil.copy(src, dst)

# --- Split katte- og hundebilder ---
split_class_images(source_cat, "cat")
split_class_images(source_dog, "dog")

print("Dataset delt i train_group1.1, train_group2.2 og validation.2 med riktig mappe-struktur.")
