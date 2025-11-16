

# Machine Learning Image Classification Projects

This repository contains multiple image classification projects implemented using Python and machine learning/deep learning techniques. Each project includes a Jupyter Notebook for training and evaluation, along with an explanation file for understanding the workflow.

## Projects Included

### 1. Fashion MNIST Classifier

* **Notebook:** `FashionMnistClassifier.ipynb`
* **Explanation:** `FashionMnistClassifierExplaination.md`
* **Description:** A deep learning model to classify fashion items (like shirts, shoes, bags) from the [Fashion MNIST dataset]. The dataset contains 28x28 grayscale images in 10 categories.
* **Key Highlights:**

  * Data preprocessing and normalization
  * Model building using neural networks
  * Evaluation with accuracy metrics
  * Visualization of predictions

### 2. Fruits Classifier

* **Notebook:** `FruitsClassifier.ipynb`
* **Explanation:** `FruitsClassifierExplaination.md`
* **Description:** A classifier for different types of fruit images.
* **Key Highlights:**

  * Data augmentation for improving performance
  * Convolutional Neural Network (CNN) architecture
  * Training, validation, and testing pipeline
  * Prediction examples

### 3. Mood Binary Classifier

* **Notebook:** `MoodBinaryClassifier.ipynb`
* **Explanation:** `MoodBinaryClassifierExplaination.md`
* **Frontend Screenshot:** `MoodBinaryClassifierFrontendImg.png`
* **Description:** A binary classification model that predicts mood (happy/not happy) from uploaded images.
* **Key Highlights:**

  * Image preprocessing for user uploads
  * Binary CNN classifier
  * Streamlit interface for real-time predictions
  * Easy-to-use front-end

## How to Run

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *(Make sure to have `tensorflow`, `numpy`, `opencv-python`, `streamlit` installed)*
3. Open the Jupyter Notebooks to train and test the models.
4. For Mood Binary Classifier frontend:

   ```bash
   streamlit run MoodBinaryClassifier.ipynb
   ```

## Repository Structure

```
├── FashionMnistClassifier.ipynb
├── FashionMnistClassifierExplaination.md
├── FruitsClassifier.ipynb
├── FruitsClassifierExplaination.md
├── MoodBinaryClassifier.ipynb
├── MoodBinaryClassifierExplaination.md
├── MoodBinaryClassifierFrontendImg.png
└── README.md
```

