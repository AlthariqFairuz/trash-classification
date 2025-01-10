# Trash Classification using CNN

This project implements a Convolutional Neural Network (CNN) model for automated trash classification using the TrashNet dataset. You can get the dataset [here](https://huggingface.co/datasets/garythung/trashnet).

## 📋 Overview

The project uses a CNN architecture to classify images of trash into different categories:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The model is trained on the TrashNet dataset from Hugging Face, achieving robust classification performance through deep learning techniques.

## 🛠️ Requirements

The project requires the following main dependencies:
- Python 3.x
- PyTorch (for neural network implementation)
- Weights & Biases (for experiment tracking)
- Matplotlib and Seaborn (for visualization)
- scikit-learn (for evaluation metrics)
- Jupyter Notebook (for interactive development)

A complete list of dependencies can be found in `requirements.txt`.

## ⚙️ Setup and Configuration

Before running the project, you'll need to set up authentication for necessary services:

### Weights & Biases Setup
1. Create an account at [Weights & Biases](https://wandb.ai)
2. Obtain your API key from the [authorization page](https://wandb.ai/authorize)
3. Set up your API key in your environment:
   ```bash
   WANDB_API_KEY='your-api-key'
   ```
#### However, it's `HIGHLY RECOMMENDED` to run the notebook in google colab in order to get the model instead of running it in local

🚀 Getting Started

1. Clone the repository:
  ```bash
    git clone https://github.com/AlthariqFairuz/trash-classification.git
    cd trash-classification
  ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run model training:
 ```bash
  python src/train.py
  ```

4. Evaluate model performance:
  ```bash
    python src/evaluate.py
  ```


## 🤖 Model Architecture
The CNN architecture consists of strategically designed layers:

  Input Layer: Accepts RGB images of trash items
  
  Convolutional Layers:

  - Three conv layers with increasing channels (16, 32, 64)
  - ReLU activation functions
  - MaxPooling layers (2x2) for dimension reduction

  Fully Connected Layers:
  
  - Flattened layer
  - Dropout layers (0.25) for regularization
  - Dense layer with 512 neurons
  - Output layer with 6 neurons (one per class)



## 📊 Model Training and Evaluation
The model is trained using:

- Cross-Entropy Loss function
- Adam optimizer
- Learning rate: 0.0001
- Batch size: 32
- Number of epochs: 10

Training progress and metrics are tracked using Weights & Biases, including:

- Training and validation accuracy
- Loss curves
- Confusion matrices
- Per-class performance metrics

## 📈 Performance Tracking
You can monitor the model's performance in real-time through [here](https://wandb.ai/althariqfairuz273-institut-teknologi-bandung/cnn-trash-classicifations?nw=nwuseralthariqfairuz273).
