import cv2
import os
import numpy as np

# ================================
# 🔴 CHANGE ONLY THIS PATH 🔴
# ================================
# Folder where your PROCESSED images are stored
# Example: face_processed folder
BASE_FOLDER = r"C:\Users\surec\OneDrive\Desktop\Project"


data = []     # stores image vectors
labels = []   # stores person IDs

label = 0     # person counter

# Loop through each person folder
for person in os.listdir(BASE_FOLDER):
    person_path = os.path.join(BASE_FOLDER, person)

    # Skip if not a folder
    if not os.path.isdir(person_path):
        continue

    # Loop through each image of that person
    for img_name in os.listdir(person_path):

        # OPTIONAL: process only image files
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(person_path, img_name)

        # Read image in grayscale
        img = cv2.imread(img_path, 0)

        if img is None:
            print("❌ Skipped (cannot read):", img_path)
            continue

        # Convert image to vector
        vector = img.flatten()

        data.append(vector)
        labels.append(label)

    label += 1

# Convert list to NumPy array
Face_DB = np.array(data).T   # Shape: (mn × p)

print("✅ DATABASE CREATED SUCCESSFULLY")
print("Face_DB shape :", Face_DB.shape)
print("Total images  :", len(labels))
print("Total persons:", label)
