import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# =================================================
# 🔴 CHANGE ONLY THIS PATH
# =================================================
BASE_FOLDER =  r"C:\Users\surec\OneDrive\Desktop\Project"

# =================================================

# ---------------------------------
# STEP 1: CREATE FACE DATABASE
# ---------------------------------
data = []
labels = []
label = 0

for person in os.listdir(BASE_FOLDER):
    person_path = os.path.join(BASE_FOLDER, person)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, 0)

        if img is None:
            continue

        data.append(img.flatten())
        labels.append(label)

    label += 1

Face_DB = np.array(data).T  # (mn × p)
labels = np.array(labels)

print("STEP 1 DONE: Database created")
print("Face_DB shape:", Face_DB.shape)

# ---------------------------------
# STEP 2: MEAN FACE
# ---------------------------------
mean_face = np.mean(Face_DB, axis=1).reshape(-1, 1)

print("\nSTEP 2 DONE: Mean face calculated")
print("Mean face shape:", mean_face.shape)

# ---------------------------------
# STEP 3: MEAN-ZERO DATA
# ---------------------------------
Delta = Face_DB - mean_face

print("\nSTEP 3 DONE: Mean-zero data created")
print("Delta shape:", Delta.shape)

# ---------------------------------
# STEP 4: COVARIANCE MATRIX
# ---------------------------------
C = np.dot(Delta.T, Delta)

print("\nSTEP 4 DONE: Covariance matrix created")
print("Covariance matrix shape:", C.shape)

# ---------------------------------
# STEP 4 (CONT.): EIGEN DECOMPOSITION
# ---------------------------------
eigenvalues, eigenvectors = np.linalg.eigh(C)

# Sort eigenvalues in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]

# ---------------------------------
# STEP 5: SELECT TOP-k EIGENVECTORS
# ---------------------------------
k = 70  # you can change this (10, 20, 30, 40,)
Psi = eigenvectors[:, :k]

print("\nSTEP 5 DONE: Top-k eigenvectors selected")
print("Psi shape:", Psi.shape)

# ---------------------------------
# STEP 6: GENERATE EIGENFACES
# ---------------------------------
Eigenfaces = np.dot(Psi.T, Delta.T)

print("\nSTEP 6 DONE: Eigenfaces generated")
print("Eigenfaces shape:", Eigenfaces.shape)

# ---------------------------------
# STEP 7: GENERATE FACE SIGNATURES
# ---------------------------------
Weights = np.dot(Eigenfaces, Delta)

print("\nSTEP 7 DONE: Face signatures generated")
print("Weights shape:", Weights.shape)

# ---------------------------------
# STEP 8: ANN TRAINING & TESTING
# ---------------------------------
X = Weights.T
y = labels

# 60% training, 40% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

ann = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=500,
    random_state=42
)

ann.fit(X_train, y_train)
print("\nSTEP 8 DONE: ANN trained")

# Test ANN
y_pred = ann.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("ANN Classification Accuracy:", accuracy * 100, "%")
