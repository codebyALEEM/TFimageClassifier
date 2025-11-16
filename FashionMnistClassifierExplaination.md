
---

# 👗 Fashion MNIST Classification

### **A Complete Step-By-Step Explanation of the Fashion MNIST CNN Model**

This notebook builds a **multi-class image classification model** using  **TensorFlow** , classifying 28×28 grayscale images of fashion items into 10 categories.

Each cell is explained so that **beginners** can understand what is happening at every step.

---

## 🖥️ **Cell 1 — Mount Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Explanation:**

* Imports the Google Colab drive module.
* `drive.mount('/content/drive')` → Mounts your Google Drive to Colab so you can access files stored there.
* You will need to authorize your Google account.
* Once mounted, you can read/write files from your Drive like a local folder.

---

## 📦 **Cell 2 — Import Required Libraries**

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
```

**Explanation:**

* `tensorflow` → Library for building and training neural networks.
* `numpy` → Library for numerical operations (arrays, matrices).
* `matplotlib.pyplot` → Library for plotting graphs and images.
* `print(tf.__version__)` → Prints TensorFlow version to confirm installation.

---

## 🔢 **Cell 3 — Check Shape of Training Images**

```python
train_images.shape
```

**Explanation:**

* Returns dimensions of the training images array.
* Fashion MNIST training set shape: `(60000, 28, 28)`
  * `60000` → Number of training images
  * `28x28` → Height and width of each image

---

## 🔢 **Cell 4 — Check Shape of Test Images**

```python
test_images.shape
```

**Explanation:**

* Returns dimensions of the test images array.
* Fashion MNIST test set shape: `(10000, 28, 28)`
  * `10000` → Number of test images
  * `28x28` → Image size

---

## 🖼️ **Cell 5 — Display a Sample Image**

```python
plt.figure()
plt.imshow(train_images[15])
plt.colorbar()
plt.grid(False)
plt.show()
```

**Explanation:**

* `plt.figure()` → Creates a new plot figure.
* `plt.imshow(train_images[15])` → Displays the 16th image in the training set.
* `plt.colorbar()` → Adds a color scale for pixel intensity.
* `plt.grid(False)` → Removes grid lines.
* `plt.show()` → Renders the plot.

---

## 🧽 **Cell 6 — Normalize Pixel Values**

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

**Explanation:**

* Pixel values range from 0 to 255.
* Dividing by 255.0 scales them to `[0, 1]` → helps the model train faster and more accurately.

---

## 🖼️ **Cell 7 — Visualize Multiple Images with Labels**

```python
plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```

**Explanation:**

* Creates a 5×5 grid to visualize 20 images.
* `cmap=plt.cm.binary` → Shows images in grayscale.
* Labels displayed below each image using `class_names`.
* Helps beginners connect image data with their corresponding labels.

---

## 🔄 **Cell 8 — Reshape Images for CNN Input**

```python
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
```

**Explanation:**

* Adds a channel dimension (`1`) for grayscale images.
* `astype('float32')` → Ensures compatibility with TensorFlow.
* CNN expects shape `(num_images, height, width, channels)`.

---

## 🧠 **Cell 9 — Build the CNN Model**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10,activation='softmax'),
])
```

**Explanation:**

* `Conv2D` → Detects features in the image.
* `MaxPooling2D` → Reduces image size, keeps important features.
* `Flatten` → Converts 2D feature maps to 1D vector.
* `Dense` → Fully connected layers for classification.
* `Dropout` → Prevents overfitting.
* `Dense(10, activation='softmax')` → Output layer for 10 classes.

---

## ⚙️ **Cell 10 — Compile the Model**

```python
model.compile(
    optimizer='rmsprop',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

**Explanation:**

* `optimizer='rmsprop'` → Adaptive learning optimizer.
* `loss=SparseCategoricalCrossentropy(from_logits=True)` → Appropriate for multi-class integer labels.
* `metrics=['accuracy']` → Tracks accuracy during training.

---

## 🏁 **Cell 11 — Evaluate the Model on Test Data**

```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('\nTest accuracy:', test_acc)
```

**Explanation:**

* `model.evaluate` → Measures performance on test data.
* `test_acc` → Shows how well the model generalizes to unseen images.

---
