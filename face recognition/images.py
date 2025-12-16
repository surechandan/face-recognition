import cv2
import os

# INPUT and OUTPUT folders
input_folder = r"C:\Users\surec\Downloads\dataset\dataset\faces"
output_folder = r"C:\Users\surec\OneDrive\Desktop\Project"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Loop through each person
for person in os.listdir(input_folder):
    person_input_path = os.path.join(input_folder, person)
    person_output_path = os.path.join(output_folder, person)

    # Create person folder in output
    os.makedirs(person_output_path, exist_ok=True)

    # Loop through each image of the person
    for img_name in os.listdir(person_input_path):
        img_path = os.path.join(person_input_path, img_name)

        # Read image in grayscale
        img = cv2.imread(img_path, 0)

        # Skip if image not loaded
        if img is None:
            print("Skipped:", img_path)
            continue

        # Resize image
        resized = cv2.resize(img, (100, 100))

        # Save image
        save_path = os.path.join(person_output_path, img_name)
        cv2.imwrite(save_path, resized)

        print("Saved:", save_path)

print("✅ All images processed successfully")