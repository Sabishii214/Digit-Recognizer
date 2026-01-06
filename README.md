
# Handwritten Digit Recognition using CNN (PyTorch)

This project implements a Convolutional Neural Network (CNN) in PyTorch to recognize handwritten digits from the Kaggle Digit Recognizer (MNIST) dataset.
The model achieves ~99.88% validation accuracy and generates a submission file compatible with Kaggle.


## Dataset
Source: Kaggle – Digit Recognizer
- train.csv — 42,000 labeled images
- test.csv — 28,000 unlabeled images
- Image size: 28 × 28 grayscale
- Classes: 10 digits (0–9)
## Install dependencies

1. Clone the repository
```bash
git clone https://github.com/your-username/digit-recognizer-cnn.git
cd digit-recognizer-cnn
```
2. Install dependencies
```bash
  $ pip install torch torchvision numpy pandas matplotlib scikit-learn
```
3. Run training script or notebook
4. Upload submission.csv to Kaggle(Optional)

## Model Architecture
The CNN architecture consists of:
- Conv2D (1 → 32) + ReLU + MaxPool
- Conv2D (32 → 64) + ReLU
- Conv2D (64 → 128) + ReLU + MaxPool
- Fully Connected Layer (128 × 7 × 7 → 128)
- Dropout (0.5)
- Output Layer (128 → 10)
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
- Epoch Stopped = 80