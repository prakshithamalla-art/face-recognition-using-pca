# PCA Face Recognition System

## Overview

This project implements a Face Recognition System using Principal Component Analysis (PCA).
PCA is used to reduce the dimensionality of face images and extract important features called **Eigenfaces**.
The system then compares a test image with training images and identifies the closest match using **Euclidean distance**.

## Objectives

* Reduce image dimensionality using PCA
* Extract important facial features (Eigenfaces)
* Project test images into PCA space
* Identify the closest matching face using Euclidean distance

## Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-learn
* Matplotlib

## Project Structure
* train  – contains training face images
* test – contains the test image
* pca_face.py – main Python program for PCA face recognition
* PCA-BASED FACE RECOGNITION SYSTEM.pdf – project report


## How It Works

1. Load training face images.
2. Convert images to grayscale and resize them.
3. Apply PCA to extract Eigenfaces.
4. Project the test image into PCA space.
5. Calculate Euclidean distance between the test image and training images.
6. The image with the smallest distance is the closest match.

## Output

The program displays:

* The test image
* The matched training image

## Author

Prakshitha Malla
