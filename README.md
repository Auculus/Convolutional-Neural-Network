# Convolutional-Neural-Network
## Overview
This project implements a **Convolutional Neural Network from scratch** for image classification on the **Fashion-MNIST dataset**. The objective was to understand and build the complete learning pipeline without relying on high-level deep learning frameworks, focusing on the internal mechanics of convolution, pooling, and backpropagation.

## Dataset
The model is trained and evaluated on the **Fashion-MNIST** dataset, which consists of 28×28 grayscale images across 10 clothing categories. The dataset serves as a benchmark for image recognition tasks while remaining computationally lightweight.

## Network Architecture
The CNN architecture is composed of the following layers:
- **Convolution Layer** – Extracts spatial features using learnable filters  
- **Max Pooling Layer** – Reduces spatial dimensionality while preserving salient features  
- **Fully Connected Layer** – Performs classification based on extracted features 

## Training Process
- Forward propagation through convolution, pooling, and dense layers  
- Loss computation using classification error  
- Backpropagation for weight and bias updates  
- Iterative training over the dataset to improve accuracy

## Future Improvements
- Add support for additional convolutional layers  
- Experiment with different activation functions and optimizers  
- Visualize feature maps and learned filters  
- Extend the model to other image datasets 
