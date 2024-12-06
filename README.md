# Dog Breed Classification Project – "Find the Samoyed"

## Description
This project is designed to create a tool for analyzing, labeling, and classifying images using machine learning and deep learning techniques. The tool processes at least 300 images, evaluates their quality, identifies and labels selected object classes, and prepares the data for further classification using the PyTorch library.

### Key Features:
- **Object Labeling (`labeling.py`)**: Detects and labels selected object classes in images (e.g., dog breeds).
- **Data Preparation for PyTorch (`custom_dataset.py`)**: Converts the dataset into a format compatible with PyTorch Dataset, facilitating training.
- **Image Classification**: Utilizes pre-trained PyTorch models and optionally trains custom models for object classification.

## Technologies
- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Image Processing Libraries**: OpenCV, Pillow, PIL (Python Imaging Library)
- **Other Libraries**: NumPy, TensorFlow (for object detection)

## Models Used:
- **ResNet50**: A deep residual network model that excels in recognizing complex objects, learning from previous mistakes. Achieved an accuracy of 93.33%.
- **AlexNet**: A classic CNN model that identifies basic shapes like lines and textures. It uses error-based learning for better classification. Accuracy: 63.33%.
- **VGG16**: A deep convolutional neural network used for recognizing various objects, especially complex ones. Accuracy: 33.33%.
- **SqueezeNet**: A lightweight model optimized for resource-constrained environments, such as mobile devices. Accuracy: 63.33%.
- **DenseNet**: A model known for its efficient layer-to-layer information sharing, suitable for models with fewer parameters. Accuracy: 33.33%.

## Results:
- **Best Model**: ResNet50, achieving an accuracy of **93.33%**.
- **Other Models' Performance**:
  - AlexNet and Simple CNN: **63.33%**
  - VGG16 and DenseNet: **33.33%**
  - SqueezeNet: **63.33%**

## Setup Instructions

To run this project locally, please follow the steps below.

### Prerequisites:
Ensure you have the following installed:
- **Python 3.6 or higher**
- **pip** (Python package installer)

You can install the necessary Python packages using `requirements.txt`:
```bash
pip install -r requirements.txt
