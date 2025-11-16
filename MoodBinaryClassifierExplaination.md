
# 🧠 Mood Classification (Happy / Not Happy)

### **A Complete Step-By-Step Explanation of the Binary Image Classification Model**

This notebook builds a **binary image classification model** using  **TensorFlow** , classifying images into:

* **Happy 😀**
* **Not Happy 😐**

Every cell is explained so that even a **beginner** can understand what's happening in the code.

---

## 🖥️ **Cell 1 — Check GPU Availability**

```python
!nvidia-smi
```

This command shows if Google Colab has given us a  **GPU** .

A GPU makes deep learning  **faster** .

---

## 📦 **Cell 2 — Import Required Libraries**

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
```

**What these do:**

* `ImageDataGenerator` – Loads images from folders & prepares them for training.
* `image` – Helps load individual images.
* `matplotlib` – To display images.
* `tensorflow` – To build the deep learning model.
* `numpy` – For numerical operations.
* `cv2` – For reading images.
* `os` – To work with folders.

---

## 🖼️ **Cell 3 — Load a Sample Image**

```python
img = image.load_img('/content/drive/MyDrive/training/happy/k3.jpg')
```

Loads an image from the **happy** folder to check where we are working.

---

## 🖼️ **Cell 4 — Display the Loaded Image**

```python
plt.imshow(img)
```

Shows the image you loaded — useful for verification.

---

## 🧪 **Cell 5 — Read Image Using OpenCV**

```python
i1 = cv2.imread('/content/drive/MyDrive/training/happy/k3.jpg')
i1
```

Reads the same image using OpenCV.

OpenCV loads images as  **NumPy arrays** .

---

## 📏 **Cell 6 — Check Image Shape**

```python
i1.shape
```

Shows  **height, width, channels** , e.g.:

`(200, 200, 3)`

(3 = RGB channels)

---

## 🧽 **Cell 7 — Create Training & Validation Generators**

```python
train  = ImageDataGenerator(rescale=1/200)
validation = ImageDataGenerator(rescale=1/200)
```

`rescale=1/200` reduces pixel values from (0–255) to (0–1.27).

Scaling helps the neural network learn better.

---

## 📚 **Cell 8 — Load Training & Validation Image Folders**

```python
train_dataset = train.flow_from_directory('/content/drive/MyDrive/training',
                                          target_size = (200,200),
                                          batch_size = 32,
                                          class_mode = 'binary')

validation = validation.flow_from_directory('/content/drive/MyDrive/validation',
                                            target_size = (200,200),
                                            batch_size = 32,
                                            class_mode = 'binary')
```

This automatically:

* Reads images from your folder structure
* Converts them to 200×200
* Bundles them in batches of 32
* Labels them as **0** or **1** (Binary)

Folder structure expected:

```
training/
    happy/
    nothappy/
validation/
    happy/
    nothappy/
```

---

## 🏷️ **Cell 9 — See Class Labels**

```python
train_dataset.class_indices
```

Shows which class is **0** and which is  **1** . Example:

```
{'happy': 0, 'nothappy': 1}
```

---

## 🔢 **Cell 10 — Numerical Class Values**

```python
train_dataset.classes
```

Lists all class labels assigned to each image.

---

## 🧠 **Cell 11 — Build the CNN Model**

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Layer-by-layer explanation:

#### 📌 **1. Convolution Layers**

Detects patterns like:

* Eyes
* Mouth
* Face structure

#### 📌 **2. MaxPooling**

Reduces image size → faster training.

#### 📌 **3. Flatten**

Converts 2D features to a single long vector.

#### 📌 **4. Dense(512)**

Learns patterns like:

* Smile shape
* Lip curve
* Eye squeeze

#### 📌 **5. Output Layer**

`Dense(1, activation='sigmoid')`

* Outputs a value between **0 and 1**
* If **< 0.5 → Happy**
* If **≥ 0.5 → Not Happy**

---

## ⚙️ **Cell 12 — Compile the CNN Model**

```python
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)
```

 **Loss** : Best for binary classification

 **Optimizer** : RMSprop (good for small datasets)

 **Metric** : Accuracy

---

## 🏋️ **Cell 13 — Train the Model**

```python
model_fit = model.fit(train_dataset, epochs=15)
```

This:

* Uses training images
* Trains for 15 complete cycles (epochs)
* Learns to classify Happy vs Not Happy

---

## 🗂️ **Cell 14 — List Testing Folder**

```python
dir_path = '/content/drive/MyDrive/testing'
for i in os.listdir(dir_path):
  print(i)
```

Shows all files inside  **testing folder** .

---

## 👁️ **Cell 15 — Display Test Images**

```python
for i in os.listdir(dir_path):
     img = image.load_img(dir_path+'//'+i,target_size=(200,200))
     plt.imshow(img)
     plt.show()
```

Displays each test image one by one.

---

## 🔍 **Cell 16 — Predict Mood for Each Test Image**

```python
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
images = np.vstack([x])

val = model.predict(images)

if val == 0:
    print("i am happy")
else:
    print("i am not happy")
```

### How it works:

* Convert image → array
* Add batch dimension
* Pass to the model
* Model outputs **0** or **1**

---

## ⚡ **Cell 17 — GPU Info Again**

```python
!nvidia-smi
```

Checks GPU after training.

---

# 🌐 Cell 18 — Define Prediction Function for Gradio App

```python
def predict_mood(image):
    img = image.resize((200,200))
    x = np.array(img)
    x = np.expand_dims(x,axis=0)
    val = model.predict(x)[0][0]

    if val < 0.5:
        return "Happy"
    else:
        return "Not happy"
```

This function:

* Resizes the uploaded image
* Converts it to an array
* Passes it to the model
* Returns the predicted mood

---

# 🎛️ Cell 19 — Create Gradio Web App Interface

```python
iface = gr.Interface(
    fn=predict_mood,
    inputs = gr.Image(type='pil',label="Upload an Image"),
    outputs = gr.Text(label="Predict Mood"),
    title = "Mood Classification (Happy/Not happy)",
    description = "Upload an image to classify if the person is happy or not happy"
)

iface.launch()
```

This makes a **simple web app** where the user can:

* Upload ANY image
* Model returns **Happy / Not Happy**

Perfect for real-time testing!
