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

## Part 1: Environment Setup

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
---

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
---

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


## Part 3: Model Building and Compilation

After preparing and augmenting the dataset, we define a **Convolutional Neural Network (CNN)** and build deeper variants (e.g., **ResNet50**) for plant species classification.  
Both models use TensorFlow’s Keras API, with layers optimized for image feature extraction and classification accuracy.

---

### Convolutional Neural Network (CNN)

This is a simple, efficient baseline architecture for image classification.  
It includes convolutional, pooling, dropout, and dense layers.

```python
from tensorflow.keras import models, layers, regularizers

model_cnn = models.Sequential([
    image_augmentation,
    layers.Rescaling(1./255, input_shape=(512, 512, 3)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model_cnn.summary()
```

Key Notes:

Uses L2 regularization and Dropout to prevent overfitting.

Final layer uses Softmax for multi-class probability prediction.

Augmentation + Rescaling ensures better generalization.

Model Compilation - We use Adam optimizer and categorical crossentropy loss for multi-class classification.
```python
model_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

You may also include advanced metrics:
```python
from tensorflow.keras.metrics import Precision, Recall

model_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', Precision(), Recall()]
)

Model Checkpointing and Early Stopping - These callbacks prevent overfitting and preserve the best-performing model.
```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath='best_cnn_model.h5', save_best_only=True)
]
```

Train the Model
```python
history_cnn = model_cnn.fit(
    ds_train,
    validation_data=ds_val,
    epochs=30,
    callbacks=callbacks
)
```

During training, you’ll observe real-time updates of accuracy and loss curves per epoch.

Typical Output Example:
```arduino
Epoch 10/30
132/132 [==============================] - 70s 530ms/step - loss: 0.441 - accuracy: 0.889 - val_loss: 0.512 - val_accuracy: 0.872
```

Visualize Training Curves
```python
import matplotlib.pyplot as plt

acc = history_cnn.history['accuracy']
val_acc = history_cnn.history['val_accuracy']
loss = history_cnn.history['loss']
val_loss = history_cnn.history['val_loss']

plt.figure(figsize=(10, 6))
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('CNN Model Accuracy')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('CNN Model Loss')
plt.show()
```

These plots help detect underfitting or overfitting trends and confirm learning stability.

Save the Trained Model
```python
model_cnn.save("/content/gdrive/MyDrive/Plant_CNN_Model.h5")
```

The saved model can later be reloaded for testing, visualization, or integration with YOLO-based disease detection.

## Part 4: ResNet50 Transfer Learning and Ensemble Modeling

To improve feature extraction and overall accuracy, this project incorporates **transfer learning** using a pretrained **ResNet50** model.  
This approach leverages the knowledge of ImageNet-trained filters and fine-tunes them on our plant leaf dataset.

---

### 1. Load Pretrained ResNet50 Model

We use the **ResNet50** architecture (pretrained on ImageNet) with its top classification layer removed.  
This allows the model to reuse generic low- and mid-level features (edges, textures, color gradients) and specialize for plant leaves.

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

resnet_base = ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(512, 512, 3)
)

# Freeze the base model initially
resnet_base.trainable = False
```
---


### 2. Add Custom Classification Head

A new classification head is added for the 11 plant classes.

```python
model_resnet = models.Sequential([
    image_augmentation,
    resnet_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model_resnet.summary()
```
---

### 3. Compile the Model
```python
model_resnet.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
Initially, the pretrained layers are frozen to retain their learned filters, while the new layers learn plant-specific representations.
---

### 4. Train the Model (Feature Extraction Phase)
```python
history_resnet = model_resnet.fit(
    ds_train,
    validation_data=ds_val,
    epochs=10,
    callbacks=callbacks
)
```

At this stage, only the newly added dense layers are trained.
---

### 5. Fine-Tuning (Unfreeze Selected Layers)

After initial convergence, we fine-tune the deeper layers for domain adaptation.
```python
resnet_base.trainable = True

# Optionally, freeze only the first few layers
for layer in resnet_base.layers[:143]:
    layer.trainable = False

model_resnet.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # lower LR for fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_resnet_fine = model_resnet.fit(
    ds_train,
    validation_data=ds_val,
    epochs=10,
    callbacks=callbacks
)
```

Fine-tuning helps the model adapt to unique leaf texture and disease patterns found in tropical species.
---

### 6. Compare CNN and ResNet50 Performance

You can visualize both models’ validation accuracy trends side-by-side.

```python
plt.figure(figsize=(10, 6))
plt.plot(history_cnn.history['val_accuracy'], label='CNN')
plt.plot(history_resnet_fine.history['val_accuracy'], label='ResNet50')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('CNN vs ResNet50 Validation Accuracy')
plt.legend()
plt.show()
```

Typically, ResNet50 achieves 5–8% higher validation accuracy than a vanilla CNN, due to its deeper residual learning structure.
---

### 7. Ensemble Model (CNN + ResNet50)

An ensemble combines the strengths of both models for more stable predictions.
```python
import numpy as np

# Predict using both models
pred_cnn = model_cnn.predict(ds_test)
pred_resnet = model_resnet.predict(ds_test)

# Weighted average ensemble
ensemble_pred = (0.4 * pred_cnn) + (0.6 * pred_resnet)
final_labels = np.argmax(ensemble_pred, axis=1)
```

Ensembling tends to smooth out model-specific biases and improves robustness under field conditions.
---

### 8. Save Fine-Tuned ResNet50 Model
```python
model_resnet.save("/content/gdrive/MyDrive/ResNet50_Finetuned.h5")
```

Both the CNN and ResNet50 models can be exported and used for inference or integrated into the Red Palm Mite detection system.

---

## 📊 Part 5: Model Evaluation, Metrics & Visualization

After training and fine-tuning the CNN and ResNet50 models, the next step is to evaluate their performance on the **unseen test dataset**.  
This section measures quantitative metrics (Accuracy, Precision, Recall, F1-Score, Cohen’s Kappa) and provides visualization tools to interpret results.

---

### 1️.. Evaluate Model Performance on Test Set
```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Evaluate on test dataset
test_loss, test_acc = model_resnet.evaluate(ds_test)
print(f"Test Accuracy = {test_acc:.4f}")
```
---

### 2️. Generate Predictions and Metrics
```python
# Collect predictions and true labels
y_true, y_pred = [], []

for images, labels in ds_test:
    preds = model_resnet.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Generate report
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

print("Precision:", precision_score(y_true, y_pred, average='weighted'))
print("Recall:", recall_score(y_true, y_pred, average='weighted'))
print("F1-Score:", f1_score(y_true, y_pred, average='weighted'))
print("Cohen’s Kappa:", cohen_kappa_score(y_true, y_pred))
```

Cohen’s Kappa accounts for chance agreement, providing a more reliable measure than raw accuracy—especially for imbalanced datasets.

### 3. Classification Report
```python
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)
```

Example Output

                precision    recall  f1-score   support
ArecaNut            0.94       0.90      0.92       180
Avocado Tree        0.91       0.88      0.89       185
...
Palm Oil            0.96       0.94      0.95       190
accuracy                                0.93      2106
macro avg           0.93       0.92      0.93      2106
weighted avg        0.93       0.93      0.93      2106

### 4. Confusion Matrix Visualization
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=False, cmap="viridis")
plt.title("Confusion Matrix – ResNet50 Model")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
```

The confusion matrix highlights class-wise prediction performance, showing which species are most commonly confused.

### 5. Plot Accuracy & Loss Curves
```python
plt.figure(figsize=(10, 5))
plt.plot(history_resnet_fine.history['accuracy'], label='Training Accuracy')
plt.plot(history_resnet_fine.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy – ResNet50')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history_resnet_fine.history['loss'], label='Training Loss')
plt.plot(history_resnet_fine.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training vs Validation Loss – ResNet50')
plt.show()
```

These curves provide a visual check for convergence and overfitting trends.
---

### 6. Model Interpretability (Grad-CAM Visualization)

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights image regions that most influenced the model’s decision.
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

last_conv_layer = model_resnet.get_layer("conv5_block3_out")
grad_model = tf.keras.models.Model(
    [model_resnet.inputs],
    [last_conv_layer.output, model_resnet.output]
)

# Select a test image
image, label = list(ds_test.take(1))[0]
image = image[0:1]  # single batch

# Compute gradients
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(image)
    class_idx = tf.argmax(predictions[0])
    loss = predictions[:, class_idx]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_outputs = conv_outputs[0]

# Weight channels by gradients
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

plt.imshow(image[0].numpy().astype("uint8"))
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.title("Grad-CAM Heatmap – ResNet50 Prediction")
plt.axis("off")
plt.show()
```
Grad-CAM visualizes which leaf areas contribute most to the classification, providing interpretability for researchers and agronomists.
