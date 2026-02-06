# AI Pneumonia X-Ray Diagnostics

**Deep Learning Based Chest X-Ray Pneumonia Detection System**

This project uses Deep Learning (CNN + Transfer Learning) models to
automatically detect Pneumonia from Chest X-Ray images. The system
compares three powerful pretrained architectures: ResNet-50,
EfficientNet, and DenseNet.

------------------------------------------------------------------------

## Problem Statement

Manual analysis of chest X-rays is time-consuming, prone to human error,
and difficult in high-patient-load hospitals. This project provides an
AI-based automated diagnostic solution.

------------------------------------------------------------------------

## Objectives

-   Detect Pneumonia from chest X-ray images
-   Train and evaluate multiple deep learning models
-   Compare performance and select the best model
-   Provide a medical decision-support tool

------------------------------------------------------------------------

## Dataset

-   Chest X-Ray Pneumonia Dataset
-   Classes: Normal and Pneumonia
-   Split: Training, Validation, Testing
-   Preprocessing: Resize, Normalization, Data Augmentation

------------------------------------------------------------------------

## System Architecture

Chest X-Ray Image → Preprocessing → Deep Learning Model → Prediction
Output

------------------------------------------------------------------------

## Models Used

### ResNet-50

Skip connections, strong baseline.

### EfficientNet (Best Model)

High accuracy with fewer parameters.

### DenseNet-121

Dense connectivity, effective for medical imaging.

------------------------------------------------------------------------

## Model Performance

  Model          Accuracy
  -------------- ----------
  ResNet-50      91.8%
  DenseNet       93.4%
  EfficientNet   94.6%

Best Performing Model: EfficientNet

------------------------------------------------------------------------

## Technologies Used

Python, TensorFlow/Keras/PyTorch, OpenCV, NumPy, Matplotlib, Transfer
Learning

------------------------------------------------------------------------

## How to Run

pip install -r requirements.txt, python app.py

------------------------------------------------------------------------

## Conclusion

EfficientNet achieved the best performance and proves AI can help
pneumonia diagnosis.

------------------------------------------------------------------------
