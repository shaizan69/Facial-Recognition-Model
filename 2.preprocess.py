import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Function to load and preprocess images
def load_and_preprocess_data(dataset_path, img_size=(150, 150)):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    # Loop through each person's folder
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            label_dict[current_label] = person_name  # Map label to person's name
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                image = cv2.imread(image_path)  # Read the image
                if image is None:
                    print(f"Warning: Unable to read image {image_path}. Skipping.")
                    continue
                image = cv2.resize(image, img_size)  # Resize image
                image = image / 255.0  # Normalize pixel values
                images.append(image)
                labels.append(current_label)
            current_label += 1

    return np.array(images), np.array(labels), label_dict

# Load and preprocess dataset
dataset_path = "dataset"  # Path to the dataset folder
images, labels, label_dict = load_and_preprocess_data(dataset_path)

# Check if dataset is loaded correctly
if len(images) == 0:
    raise ValueError("No images found in the dataset. Check the dataset path and structure.")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
num_classes = len(label_dict)  # Number of unique people
y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

print("Dataset loaded and preprocessed.")
print(f"Number of classes: {num_classes}")
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train_one_hot.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Testing labels shape: {y_test_one_hot.shape}")