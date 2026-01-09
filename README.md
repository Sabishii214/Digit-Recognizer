
# Handwritten Digit Recognition using CNN (PyTorch)

This project implements a Convolutional Neural Network (CNN) in PyTorch to recognize handwritten digits. It includes a training pipeline using the Kaggle MNIST dataset and an interactive web-based interface for real-time digit recognition.
The model achieves ~99.88% validation accuracy.

## Features
- Training Pipeline: High-accuracy CNN training with Early Stopping.
- Interactive UI: A Gradio-based web interface where you can draw digits and get instant predictions.
- Kaggle Ready: Generates a submission.csv for Kaggle competition entry.

## Model Architecture
- Conv2D (1 → 32, 3x3) + ReLU + MaxPool
- Conv2D (32 → 64, 3x3) + ReLU
- Conv2D (64 → 128, 3x3) + ReLU + MaxPool
- Fully Connected (128 × 7 × 7 → 128) + ReLU
- Dropout (0.5)
- Output Layer (128 → 10)

## Dataset
Source: Kaggle – Digit Recognizer
- train.csv — 42,000 labeled images
- test.csv — 28,000 unlabeled images
- Image size: 28 × 28 grayscale
- Classes: 10 digits (0–9)

## Install dependencies

1. Clone the repository
```bash
git clone https://github.com/Sabishii214/Digit-Recognizer.git
cd Digit-Recognizer
```
2. Install dependencies
```bash
  $ pip install torch torchvision numpy pandas matplotlib scikit-learn gradio Pillow
```
## Usage

1. Training
Run the Jupyter notebook Digit-Recognizer.ipynb to train the model. This will output a weight file named mnist_cnn.pth upon completion.

2. Interactive Prediction (Gradio UI)
Once you have the mnist_cnn.pth file in your project folder, run the interactive sketchpad interface:
```bash
  python Predict.py
  ```
- This will launch a local web server (usually at http://127.0.0.1:7860).
- Open the link in your browser.
- Draw a digit on the canvas, and the model will display the top 3 most likely predictions.

## Training details
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: CrossEntropyLoss
- Batch Size: 64
- Epochs: Up to 100
- Early Stopping:
- Patience = 10 epochs
- Stops when validation loss stops improving

## Results

- Validation Accuracy = 99.88%
- Validation Loss	~ 0.0018
- Epoch Stopped = 80 (Stopped via Early Stopping)

## File Structure
- digit-recognizer.ipynb: Data processing, training, and evaluation.
- Predict.py: Gradio interface script for real-time testing.
- mnist_cnn.pth: Trained model weights (generated after training).