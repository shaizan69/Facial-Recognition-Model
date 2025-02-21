import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load and preprocess images
def load_and_preprocess_data(dataset_path, img_size=(150, 150)):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            label_dict[current_label] = person_name
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Unable to read image {image_path}. Skipping.")
                    continue
                image = cv2.resize(image, img_size)
                image = image / 255.0  # Normalize pixel values
                images.append(image)
                labels.append(current_label)
            current_label += 1

    return np.array(images), np.array(labels), label_dict

# Load and preprocess dataset
dataset_path = "dataset"
images, labels, label_dict = load_and_preprocess_data(dataset_path)

# Check if dataset is loaded correctly
if len(images) == 0:
    raise ValueError("No images found in the dataset. Check the dataset path and structure.")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
num_classes = len(label_dict)
y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

# Ensure X_train and y_train_one_hot are NumPy arrays
X_train = np.array(X_train)
y_train_one_hot = np.array(y_train_one_hot)

# Check their shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train_one_hot shape: {y_train_one_hot.shape}")

# Define the CNN model
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Build and compile the model
input_shape = (150, 150, 3)
num_classes = len(label_dict)
model = build_cnn_model(input_shape, num_classes)

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Print model summary
model.summary()

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Train the model with data augmentation
history = model.fit(datagen.flow(X_train, y_train_one_hot, batch_size=32),
                    epochs=30,  # Increase epochs
                    validation_data=(X_test, y_test_one_hot))

# Save the model
model.save("face_recognition_cnn_improved.h5")
print("Model trained and saved.")