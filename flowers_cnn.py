import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import tensorflow as tf
import pathlib
from sklearn.model_selection import train_test_split

# --- 1. LOAD AND PREPROCESS DATA ---

# Download and extract the dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# The get_file function returns the path to the directory where the data is cached.
# The actual image folders are inside a 'flower_photos' subdirectory.
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, cache_dir='.', untar=True)
data_dir = pathlib.Path(data_dir) / 'flower_photos'

# Define class names and their corresponding labels
flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}

# Create a dictionary of image paths for each class
flowers_images_dict = {
    flower_name: list(data_dir.glob(f'{flower_name}/*'))
    for flower_name in flowers_labels_dict.keys()
}

# Load images and labels into arrays
X, y = [], []
for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        # Resize all images to a consistent size
        resized_img = cv2.resize(img, (180, 180))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Normalize pixel values from 0-255 to 0-1
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0


# --- 2. DEFINE THE CNN MODEL WITH DATA AUGMENTATION ---

# Define data augmentation layers to prevent overfitting
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(180, 180, 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Build the CNN model
model = tf.keras.Sequential([
    data_augmentation,
    # Convolutional Block 1
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    # Convolutional Block 2
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    # Convolutional Block 3
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    # Flatten and Dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer with units equal to the number of classes
    tf.keras.layers.Dense(len(flowers_labels_dict))
])


# --- 3. COMPILE AND TRAIN THE MODEL ---

# Compile the model
# Using SparseCategoricalCrossentropy because labels are integers, not one-hot encoded
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=30, 
                    validation_data=(X_test_scaled, y_test),
                    verbose=1)


# --- 4. EVALUATE AND SAVE ---

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'\nTest accuracy: {test_acc:.4f}')

# Save the model
model.save('flowers_cnn_model.h5')
print("Model saved to flowers_cnn_model.h5")


# --- 5. VISUALIZE RESULTS ---

# Plot and save training history
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('model_performance.png')
print("Performance plot saved to model_performance.png")
plt.show()

# Visualize predictions
class_names = list(flowers_labels_dict.keys())
predictions = model.predict(X_test_scaled)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i])
    
    predicted_label_index = np.argmax(predictions[i])
    true_label_index = y_test[i]
    
    predicted_class = class_names[predicted_label_index]
    true_class = class_names[true_label_index]
    
    color = 'blue' if predicted_label_index == true_label_index else 'red'
    plt.xlabel(f'{predicted_class} ({true_class})', color=color)

plt.tight_layout()
plt.savefig('predictions.png')
print("Predictions plot saved to predictions.png")
plt.show()