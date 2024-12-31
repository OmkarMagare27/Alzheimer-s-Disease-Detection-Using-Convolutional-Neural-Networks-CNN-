# Alzheimer-s-Disease-Detection-Using-Convolutional-Neural-Networks-CNN-

# Project Overview
This project focuses on developing a machine learning model to detect and classify the stages of Alzheimer’s disease using brain MRI scans. The goal is to classify patients into four stages of cognitive decline: Non-Demented, Very Mild Dementia, Mild Dementia, and Moderate Dementia.

The model is built using Convolutional Neural Networks (CNN), which are effective for image classification tasks like this one. The dataset consists of over 85,000 MRI scans from the OASIS study, providing a rich source of data to train and evaluate the model.

# Dataset
The dataset is sourced from the OASIS (Open Access Series of Imaging Studies), which includes brain MRI scans of individuals at various stages of Alzheimer’s disease. The dataset is divided into four categories:

Non-Demented: 67,200 images
Very Mild Dementia: 13,700 images
Mild Dementia: 5,002 images
Moderate Dementia: 488 images
The dataset is available at: OASIS dataset: https://sites.wustl.edu/oasisbrains.

# Project Goals
Classification: Classify the severity of Alzheimer’s disease into four stages using MRI scans.
Model Development: Build and train a CNN model using TensorFlow and Keras.
Data Augmentation: Use techniques like rotation, flipping, and zooming to handle class imbalance and improve model robustness.
Evaluation: Analyze model performance using confusion matrix, precision, recall, and F1-score.

# Technologies Used
Programming Language: Python
Deep Learning Framework: TensorFlow, Keras
Libraries: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

# Model Architecture
Two types of CNN architectures were compared:

EfficientNetB0: A pre-trained model used for feature extraction, leveraging transfer learning for fast training.
Custom CNN: A fully customized CNN architecture built from scratch to evaluate performance on this specific dataset.

# Key Layers in the Model:
Convolutional layers for feature extraction
Batch normalization for faster convergence
Global average pooling to reduce the dimensionality
Fully connected layers for classification

# Results
Training Accuracy: 97.6%
Validation Accuracy: 94.65%
Test Accuracy: 94%
The model successfully differentiated between the stages of Alzheimer's with minimal misclassification, particularly between the adjacent stages.

# Installation and Usage
Prerequisites:
Python 3.x
TensorFlow
Keras
NumPy
Matplotlib

# Contributing
Feel free to fork the repository, create a pull request, or open issues for any improvements or bug fixes!
