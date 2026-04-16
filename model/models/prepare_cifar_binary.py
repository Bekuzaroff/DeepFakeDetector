import os
import pickle
import zipfile
import numpy as np
from PIL import Image
from tqdm import tqdm

# === Modify this path ===
cifar_dir = r"C:\Users\SWARNAJIT ROY\Desktop\Projects\iee\cifar-10-batches-py"
output_dir = "cifar_binary"
zip_path = "cifar_binary.zip"

# Real and Fake labels
real_classes = {3, 4, 5, 7}  # cat, deer, dog, horse
fake_classes = {0, 1, 8, 9}  # airplane, automobile, ship, truck

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def save_images_from_batch(batch_file, split):
    data_dict = unpickle(batch_file)
    data = data_dict[b'data']
    labels = data_dict[b'labels']
    filenames = data_dict[b'filenames']

    for i in tqdm(range(len(data)), desc=f"{split}"):
        label = labels[i]
        if label in real_classes:
            class_dir = os.path.join(output_dir, split, "real")
        elif label in fake_classes:
            class_dir = os.path.join(output_dir, split, "fake")
        else:
            continue

        os.makedirs(class_dir, exist_ok=True)
        img = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)
        filename = filenames[i].decode('utf-8')
        img.save(os.path.join(class_dir, filename))

# Process batches
for i in range(1, 5):  # First 4 batches as training
    save_images_from_batch(os.path.join(cifar_dir, f"data_batch_{i}"), 'train')

# Batch 5 as validation
save_images_from_batch(os.path.join(cifar_dir, "data_batch_5"), 'val')

# Test batch
save_images_from_batch(os.path.join(cifar_dir, "test_batch"), 'test')

# Zip it all
print("Zipping the dataset...")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, output_dir)
            zipf.write(file_path, arcname)

print(f"✅ Dataset ready and zipped at: {os.path.abspath(zip_path)}")
