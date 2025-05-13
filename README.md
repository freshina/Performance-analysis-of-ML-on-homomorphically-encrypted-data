# Performance Analysis of Machine Learning on Homomorphically Encrypted Data

This repository contains code and resources related to the project on machine learning with homomorphic encryption, as described in our published paper.

## Publications

For a detailed analysis of our work, please refer to our published paper:
- [Performance Analysis of Machine Learning on Homomorphically Encrypted Data](https://ieeexplore.ieee.org/abstract/document/10826872/) - Published in IEEE Xplore, 2024.

## Machine Learning Model Collection

This repository contains a collection of machine learning models implemented in Python, focusing on various datasets such as Pima, Titanic, fraud detection, and MNIST. Each directory is dedicated to a specific dataset or problem, containing the necessary code and data files.

## Table of Contents

1. [Pima Diabetes](#pima-diabetes)
2. [Titanic Survival Prediction](#titanic-survival-prediction)
3. [Fraud Detection](#fraud-detection)
4. [MNIST Handwritten Digits Recognition](#mnist-handwritten-digits-recognition)

## Pima Diabetes

### Description
The Pima Diabetes directory contains a project that predicts the onset of diabetes within five years based on diagnostic measures.

### Files
- `diabetes.csv`: The dataset used for training and testing the model.
- `pima.py`: The Python script containing the code for the diabetes prediction model.

## Titanic Survival Prediction

### Description
The Titanic directory contains a project that predicts the survival of passengers on the Titanic.

### Files
- `Titanic.py`: The Python script for the Titanic survival prediction model.
- `gender_submission.csv`: A sample submission file for the competition.
- `test.csv`: The test dataset.
- `titanic.csv`: The training dataset.
- `train.csv`: Another version of the training dataset.

## Fraud Detection

### Description
The fraud_detection directory contains a project that identifies fraudulent transactions.

### Files
- `fraud_detection.py`: The Python script for the fraud detection model.
- `sample_submission.csv`: A sample submission file for the competition.
- `test_identity.csv`: Test data with identities (if applicable).

## MNIST Handwritten Digits Recognition

### Description
The mnist directory contains projects that recognize handwritten digits using neural networks.

### Files
- `CNN_EncCNN.py`: A Convolutional Neural Network (CNN) model for MNIST.
- `EncCNN8192.py`: Encrypted CNN model with a polynomial degree of 8192.
- `EncFCNN8192.py`: Encrypted Fully Connected Neural Network (FCNN) model with a polynomial degree of 8192.
- `FCNN_EncFCNN4096.py`: Encrypted FCNN model with a polynomial degree of 4096.
- `mnist_neural_networks.py`: A script for training neural networks on the MNIST dataset.

## Getting Started

### Requirements
- Python 3.x
- Libraries: `torch`, `torchvision`, `tenseal`, `pandas`, `sklearn`, etc.

### Installation
Follow these steps to install and set up the SEAL environment:

1. Install necessary packages:
   ```bash
   sudo apt-get install git build-essential cmake python3 python3-dev python3-pip
   
2.Clone the repository:
git clone https://github.com/Huelse/SEAL-Python.git
cd SEAL-Python

3.Install dependencies:
pip3 install numpy pybind11

4.Initialize the SEAL and pybind11 submodules:
git submodule update --init --recursive

5.Build the SEAL library without msgsl, zlib, and zstandard compression:
cd SEAL
cmake -S . -B build -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF -DSEAL_USE_ZSTD=OFF
cmake --build build
cd ..

6.Generate the dynamic library:
python3 setup.py build_ext -i

7.Usage
After setting up the environment, you can use the SEAL library in your Python projects. The dynamic library generated will be in the current directory.

8.Testing
To test the SEAL library, run the following commands:
cp seal.*.so examples
cd examples
python3 4_bgv_basics.py

9.Acknowledgments
Microsoft for developing the SEAL library.
The contributors to the SEAL-Python project for making the Python bindings available.
