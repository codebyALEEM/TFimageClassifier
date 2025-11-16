# 🍎 Fruit Classification Model (Beginner-Friendly Explanation)

This file explains **every single line** of your notebook in  **simple English** , everyone can understand what is happening.

---

# 📌 **Cell 1 — Connecting Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

### ✔ Explanation

Google Colab does not store files permanently, so we connect to  **Google Drive** .

This allows Colab to read or save files (datasets, models, etc.).

* `drive.mount()` attaches your Drive inside Colab at the folder `/content/drive`.

---

# 📌 **Cell 2 — Downloading the Fruit Dataset**

```python
!wget https://bitbucket.org/ishaanjav/code-and-deploy-custom-tensorflow-lite-model/raw/a4febbfee178324b2083e322cdead7465d6fdf95/fruits.zip
```

### ✔ Explanation

* `wget` downloads a ZIP file from the given link.
* The ZIP file contains the **fruit images** used to train the model.

---

# 📌 **Cell 3 — Unzipping the Dataset**

```python
!unzip fruits.zip
```

### ✔ Explanation

This extracts the downloaded ZIP file and creates folders like:

* `fruits/train/`
* `fruits/validation/`
* `fruits/test/`

Each folder contains images of apples, bananas, and oranges.

---

# 📌 **Cell 4 — Importing Required Libraries**

```python
import tensorflow as tf
import matplotlib.pyplot as plt
```

### ✔ Explanation

* `tensorflow` → used for building and training the machine learning model.
* `matplotlib.pyplot` → used to display images and graphs.

---

# 📌 **Cell 5 — Loading Training, Validation & Test Data**

```python
img_height, img_width = 32, 32
batch_size = 20
```

✔ Sets image size and number of images processed together.

---

### **Loading Training Data**

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    "fruits/train",
    image_size=(img_height, img_width),
    batch_size=batch_size
)
```

### ✔ Explanation

This reads all fruit images from the `train` folder and automatically labels them.

---

### **Loading Validation Data**

```python
val_ds = tf.keras.utils.image_dataset_from_directory(
    "fruits/validation",
    image_size=(img_height, img_width),
    batch_size=batch_size
)
```

✔ Used to check how well the model is learning *during* training.

---

### **Loading Test Data**

```python
test_ds = tf.keras.utils.image_dataset_from_directory(
    "fruits/test",
    image_size=(img_height, img_width),
    batch_size=batch_size
)
```

✔ Used to measure model accuracy on  *completely new images* .

---

# 📌 **Cell 6 — Displaying Sample Images**

```python
class_names = ['apple', 'banana', 'orange']
plt.figure(figsize=(10, 10))
```

✔ These are the actual fruit category names.

---

### **Showing 9 Images from Training Data**

```python
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
```

### ✔ Explanation

* The loop loads **one batch** of images.
* The next loop displays  **9 sample images** .
* The title shows the correct fruit name.

This helps visually verify that dataset is loaded correctly.

---

# 📌 **Cell 7 — Building the CNN Model**

```python
model = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1./255),
```

✔ Converts pixel values from **0–255** to  **0–1** .

---

### **Feature-Extracting Layers**

```python
        tf.keras.layers.Conv2D(32,3,activation='relu'),
        tf.keras.layers.MaxPooling2D(),
```

✔ Detects edges, colors, textures.

✔ Reduces image size while keeping important parts.

---

### **Deeper Feature Layers**

```python
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.MaxPooling2D(),
```

✔ Learns more complex patterns like shapes of apples, curves of bananas, etc.

---

### **Flatten + Dense Layers**

```python
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='softmax'),
        tf.keras.layers.Dense(3)
    ]
)
```

✔ `Flatten` turns image data into a single list.

✔ `Dense(128)` learns relationships between features.

✔ `Dense(3)` outputs 3 numbers → one for each fruit.

---

# 📌 **Cell 8 — Compiling the Model**

```python
model.compile(
    optimizer='rmsprop',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

### ✔ Explanation

* **optimizer** adjusts how the model learns
* **loss** measures mistakes
* **accuracy** tells how often predictions are correct

---

# 📌 **Cell 9 — Training the Model**

```python
model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 20
)
```

### ✔ Explanation

* `epochs = 20` → the model sees the entire dataset 20 times
* After each epoch, accuracy improves

---

# 📌 **Cell 10 — Testing the Model**

```python
model.evaluate(test_ds)
```

### ✔ Explanation

Checks accuracy using the **test dataset** which the model hasn’t seen before.

---

# 📌 **Cell 11 — Display Predictions**

```python
import numpy
```

Used for finding highest scoring class.

---

### **Loop for Displaying 9 Predictions**

```python
plt.figure(figsize=(10,10))
for images,labels in test_ds.take(1):
  classifications = model(images)
```

✔ Makes predictions for one batch of test images.

---

### **Showing Predicted vs Real Fruit**

```python
  for i in range(9):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(images[i].numpy().astype('uint8'))
    index = numpy.argmax(classifications[i])
    plt.title("Pred:"+class_names[index]+"Real:"+class_names[labels[i]])
plt.show()
```

### ✔ Explanation

* `numpy.argmax` finds which fruit has the highest score.
* Displays image with:

  **Predicted fruit vs Real fruit**

---

# 📌 **Cell 12 — Converting to TFLite**

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

✔ Converts the TensorFlow model into **TensorFlow Lite** format.

This is required for  **mobile apps / embedded devices** .

---

### **Saving the TFLite Model**

```python
with open("model.tflite","wb") as f:
  f.write(tflite_model)
```

✔ Saves the converted model as a file named  **model.tflite** .

---
