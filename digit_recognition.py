import numpy as np  
import matplotlib.pyplot as plt  
import tensorflow as tf  
from tensorflow import keras  

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Print dataset shapes
print(f"Training Data Shape: {x_train.shape}")
print(f"Testing Data Shape: {x_test.shape}")

# Display some sample images
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.axis('off')
    plt.title(f"Label: {y_train[i]}")
plt.show()

# Data Preprocessing

# Normalize pixel values (scale from 0-255 to 0-1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape images for CNN input (Adding a channel dimension)
x_train = x_train.reshape(-1, 28, 28, 1)  
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical (one-hot encoding)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Print updated shapes
print(f"New Training Data Shape: {x_train.shape}")
print(f"New Testing Data Shape: {x_test.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Build the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),  # Flatten feature maps into a vector
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout to reduce overfitting
    Dense(10, activation='softmax')  # 10 output classes (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=128)

# Save the trained model
model.save("digit_recognition_model.h5")

# Plot training history
import matplotlib.pyplot as plt

# Plot accuracy and loss
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.legend()
plt.title('Model Loss')

plt.show()


import random

# Load the saved model
from tensorflow.keras.models import load_model
model = load_model("digit_recognition_model.h5")

# Pick a random image from the test set
index = random.randint(0, len(x_test) - 1)
test_image = x_test[index].reshape(1, 28, 28, 1)  # Reshape for model input

# Predict the digit
prediction = model.predict(test_image)
predicted_label = np.argmax(prediction)  # Get the digit with highest probability

# Display the image and prediction
plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.title(f"Predicted Digit: {predicted_label}")
plt.axis('off')
plt.show()

import cv2

# Load the saved model
model = load_model("digit_recognition_model.h5")

# Load the handwritten digit image
image = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)

# Resize to 28x28 if needed
image = cv2.resize(image, (28, 28))

# Invert colors if needed (Make background black & digit white)
image = cv2.bitwise_not(image)

# Normalize & reshape for model input
image = image.astype('float32') / 255.0
image = image.reshape(1, 28, 28, 1)

# Predict the digit
prediction = model.predict(image)
predicted_label = np.argmax(prediction)

# Display result
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted Digit: {predicted_label}")
plt.axis('off')
plt.show()
