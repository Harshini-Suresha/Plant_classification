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

Output Example:

```pgsql
Found 10534 files belonging to 11 classes.
Using 8428 files for training.
Using 2106 files for validation.
```

Why Split the Data?

Splitting prevents overfitting and helps assess the model’s generalization ability on unseen data.

Training set → Used to optimize weights

Validation set → Used to tune hyperparameters

Test set → Used for final evaluation

### Batch Size and Class Labels
```python
batch_size = 64
class_names = dataset.class_names
print("Classes:", class_names)
```

Output Example:
```python
['ArecaNut', 'Avocado tree', 'Banana palm', 'Bird of Paradise',
 'Cast Iron Plant', 'Citrus tree', 'Coconut palm', 'Date palm',
 'Ginger', 'Orchid', 'Palm oil']
```

Further Split for Testing
```python
val_batches = tf.data.experimental.cardinality(ds_validation)
test_dataset = ds_validation.take(val_batches // 5)
validation_dataset = ds_validation.skip(val_batches // 5)
```

This keeps 20% of the validation set for testing.

test_dataset → Used for final accuracy evaluation

validation_dataset → Used during model training

Resize Images for Uniformity

All images are resized to 512×512 pixels for consistent tensor dimensions.

```python
size = (512, 512)

ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_val = validation_dataset.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = test_dataset.map(lambda image, label: (tf.image.resize(image, size), label))
```

This resizing step ensures all models (CNN, ResNet, etc.) receive uniformly shaped input data.

### Visualize Sample Images

Display a few samples from the dataset to verify correct loading and labeling.
```python
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 10))
for images, labels in ds_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()
```

Example Output:
A 3×3 grid of correctly labeled plant leaf images confirming dataset integrity.

### Data Augmentation

To prevent overfitting and improve robustness, random transformations are applied to training images.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

image_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(height_factor=(-0.2, -0.3),
                      width_factor=(-0.2, -0.3),
                      interpolation='bilinear'),
    layers.RandomContrast(0.1),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
], name="image")
```

Data Augmentation artificially increases dataset diversity without additional image collection.

### Visualize Augmented Images
```python
import numpy as np
plt.figure(figsize=(10, 10))

for images, labels in ds_train.take(1):
    first_image = images[0]
    for i in range(9):
        augmented_image = image_augmentation(tf.expand_dims(first_image, 0), training=True)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_image[0].numpy().astype("uint8"))
        plt.axis("off")
plt.show()
```

The model now sees rotated, flipped, zoomed, and translated versions of the same leaf image,
helping it generalize better to field conditions and variable lighting.
