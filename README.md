# face-recognition
i have completed face recognition system This project implements a face recognition system using Principal Component Analysis (PCA) and an Artificial Neural Network (ANN). PCA is used to extract discriminative facial features (Eigenfaces) and reduce dimensionality, while ANN is used for classification. 
The system is evaluated using a 60% training and 40% testing split, and performance is analyzed by varying the number of eigenfaces.
Technologies Used:
Python 3.11
OpenCV
NumPy
scikit-learn
KEY CONCEPTS USED:
Image preprocessing (grayscale, resizing)
Principal Component Analysis (PCA)
Eigenfaces
Feature extraction (face signatures)
Artificial Neural Network (ANN)
Accuracy vs k (number of eigenfaces) analysis
Imposter (not enrolled person) detection
DATASET:
Total images: 450
Image size: 100 × 100
Dataset split:
270 images (60%) for training
180 images (40%) for testing
Images are organized person-wise in folders.
METHODOLOGY:
Load and preprocess face images
Convert images into vector form
Compute mean face and mean-zero images
Apply PCA using surrogate covariance matrix
Generate Eigenfaces
Extract face signatures (PCA features)
Train ANN classifier
Test classifier and evaluate accuracy
Analyze accuracy vs k
Perform imposter detection
Conclusion
This project demonstrates a classical approach to face recognition using PCA and ANN. While PCA efficiently reduces dimensionality, accuracy depends on the choice of k and ANN parameters. The project highlights the limitations of linear models and provides a strong academic foundation for understanding face recognition systems.

