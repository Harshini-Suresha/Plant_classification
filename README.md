# Deep Learning-Based Classification of Plants and Detection of Red Palm Mite Infestation from Leaf Images

This repository implements the research work titled  
**“Deep Learning-Based Classification of Plants and Detection of Red Palm Mite Infestation from Leaf Images.”**  
The project combines **deep learning** and **machine learning** techniques to classify tropical plant species and detect symptoms of **Red Palm Mite (Raoiella indica)** infestation directly from leaf images.  
It was designed and executed in **Google Colab**, integrating TensorFlow, Keras, and Scikit-Learn frameworks for end-to-end training, evaluation, and visualization.

---

## Abstract

Red Palm Mite-affected plant classification and detection were conducted separately.  
Plant classification identified which plant belonged to which class using deep learning models including **CNN, EfficientNet, MobileNet, ViT, ResNet50, Xception, and Inception V3**, along with machine-learning classifiers such as **Random Forest, SVM, and KNN**.  
Disease detection focused on identifying symptoms such as **Healthy**, **Yellow Spots**, **Reddish Bronzing**, and **Silk Webbing** through CNN and **YOLOv8** architectures.

The dataset, sourced from *PlantVillage* and curated from open repositories (Kaggle, Mendeley, Roboflow, Manipal Forestry datasets), comprised **11 plant classes and 11 550 images**, evenly distributed across species such as *Areca Nut, Bird of Paradise, Date Palm, Cast Iron, Citrus Tree, Banana Palm, Coconut Palm, Orchid, Palm Oil, Ginger,* and *Avocado Tree.*  
Automated labeling of disease classes was achieved using **Snorkel**, which leverages heuristic rules and external knowledge to reduce manual annotation effort while improving labeling reliability.

This repository reproduces the complete workflow, including **dataset preparation**, **augmentation**, **model training**, **evaluation metrics**, and **visualization of results**, supporting reproducibility for research and applied deployment.

---

## Project Overview

| Component | Description |
|------------|--------------|
| **Plant Classification** | Uses CNN, ResNet50, EfficientNet, MobileNet, ViT, Xception, Inception V3, Random Forest, SVM, and KNN to identify plant species. |
| **Disease Detection** | Uses CNN and YOLOv8 for multi-class detection of Red Palm Mite infestation symptoms. |
| **Dataset** | 11 classes × ~1050 images each (PlantVillage + field images). |
| **Data Augmentation** | Random flip, rotation, zoom, translation, and contrast enhancement using `tf.keras.layers`. |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-score, and Cohen’s Kappa. |
| **Visualization** | Heatmaps of confusion matrices, model-training curves, and feature visualizations. |
| **Implementation Platform** | Google Colab + Google Drive integration for model and dataset management. |

---

## Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/red-palm-mite-classification.git
cd red-palm-mite-classification
```
---

### 2. Install Dependencies

All required packages are installable directly in Colab or a Python ≥ 3.10 environment:
```python
pip install tensorflow==2.15.1 tensorflow-addons tensor-dash tensorflow-datasets
pip install scikit-learn seaborn matplotlib numpy pillow
```

### 3. Connect Google Drive in Colab

Mount Google Drive to access datasets and save trained models:

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

After authorization, your Drive contents become accessible under /content/gdrive/MyDrive/.


📂 Dataset Structure

The dataset should be organized as follows:
```mathematica 
/MyDrive/Classes/
    ├── ArecaNut/
    ├── Bird of Paradise/
    ├── Cast Iron Plant/
    ├── Date Palm/
    ├── Monstera deliciosa/
    ├── Citrus tree/
    ├── Coconut palm/
    ├── Banana palm/
    ├── Orchid/
    ├── Ginger/
    └── Palm oil/
```
Each folder contains images for that class (.jpg, .jpeg, or .png).
A secondary labeled dataset in CSV format is used for disease classification (Healthy / Yellow Spots / Reddish Bronzing / Silk Webbing).

---

## Part 2: Data Loading and Pre-Processing

The first step is to load images directly from your Google Drive and create training, validation, and test datasets.  
TensorFlow’s `image_dataset_from_directory()` utility automatically infers class labels from sub-folder names.

---

### Load and Split the Dataset
```python
import tensorflow as tf

# Load all images from directory
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/gdrive/MyDrive/Classes"
)

# Split into 80% training and 20% validation
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/gdrive/MyDrive/Classes",
    validation_split=0.2,
    subset="training",
    seed=123
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/gdrive/MyDrive/Classes",
    validation_split=0.2,
    subset="validation",
    seed=123
)
```
