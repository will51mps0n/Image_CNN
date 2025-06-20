# Image_CNN

This project trains and evaluates deep learning models on the CIFAR-10 dataset. Two architectures are implemented: a fully connected network (FCN) and a convolutional neural network (CNN). The models are evaluated on classification accuracy across 10 object categories.

## Features
- CIFAR-10 classification with PyTorch
- Support for FCN and CNN models
- Saved model checkpoints included
- Grid search for activation functions and learning rates
- Training/validation accuracy output

## Technologies
- Python
- PyTorch
- Matplotlib
- NumPy

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Train and evaluate:
   python run.py
3. To switch between models, edit run.py to load either:
   cifar-10-fcn.pt
   cifar-10-cnn.pt
