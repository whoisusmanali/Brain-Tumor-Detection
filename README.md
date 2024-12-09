# Brain Tumor Detection using CNN and Transfer Learning

This project demonstrates the detection of brain tumors using Convolutional Neural Networks (CNN) and transfer learning techniques. The goal is to classify MRI brain scans into tumor and non-tumor categories with high accuracy by leveraging the power of pre-trained deep learning models and thorough data preprocessing.

---


## Features

- **Data Cleaning and Preprocessing**:  
  Comprehensive filtering, cleaning, and augmentation of the dataset to enhance model training and prevent overfitting.
  
- **Deep Learning with CNN**:  
  Utilization of Convolutional Neural Networks to automatically extract features from MRI images.

- **Transfer Learning**:  
  Fine-tuning pre-trained models (e.g., VGG16, ResNet50) to leverage learned features and improve performance on a relatively smaller dataset.

- **Visualization**:  
  Clear visualization of the dataset, training process, and results, including confusion matrix and accuracy plots.

---

## Dataset

- The dataset consists of MRI images of brain scans classified into **tumor** and **non-tumor** categories.
- Images were subjected to:
  - **Noise reduction** using filtering techniques.
  - **Normalization** to standardize pixel values.
  - **Data augmentation** to increase diversity (flipping, rotation, scaling, etc.).

---

## Methodology

1. **Data Collection**:  
   Gathered high-quality MRI scans and divided them into training, validation, and testing sets.

2. **Data Preprocessing**:  
   - Applied noise filtering and normalization.
   - Augmented data to improve generalization.

3. **Model Selection**:  
   - Implemented a custom CNN from scratch.
   - Fine-tuned pre-trained models using transfer learning (e.g., ResNet50, EfficientNet).

4. **Training**:  
   - Optimized the model with techniques such as early stopping, learning rate scheduling, and dropout layers.

5. **Evaluation**:  
   - Assessed the model using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.

---

## Model Architecture

- **Custom CNN**:  
  - Conv2D and MaxPooling layers for feature extraction.
  - Fully connected layers with softmax activation for classification.

- **Transfer Learning**:  
  - Pre-trained models as feature extractors.
  - Additional fully connected layers for the specific task of tumor detection.

---

## Results

- Achieved an accuracy of **91%** on the test dataset.
- Model demonstrated robustness across various metrics:
  - F1-Score: **88%**
- Visualizations:
  - Loss and accuracy curves.
  - Confusion matrix showing class-wise performance.

---

## Requirements

- Python 3.9+
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn
- Pandas

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```

4. Train the model:
   ```bash
   python train.py
   ```

5. Evaluate the model:
   ```bash
   python evaluate.py
   ```

---

## Future Enhancements

- Integrating more advanced pre-trained models like Vision Transformers.
- Expanding the dataset for improved generalization.
- Deploying the model as a web application for real-time predictions.
- Incorporating explainability techniques like Grad-CAM for model interpretation.

---

## Acknowledgments

Special thanks to open-source contributors and data providers for making this project possible. This project was inspired by the need to assist healthcare professionals in early and accurate detection of brain tumors.

--- 

Feel free to contribute to the project by opening issues or pull requests. Let's build impactful AI solutions together!