# Real-Time Facial Recognition using CNN

This project implements a **real-time facial recognition system** using a **Convolutional Neural Network (CNN)**. The model is trained on a dataset of facial images and can recognize individuals in real-time using a webcam. The project leverages **OpenCV** for image processing and **TensorFlow/Keras** for building and training the CNN model.

---

## Features

- **Real-Time Face Recognition**: The system can detect and recognize faces in real-time using a webcam.
- **CNN Model**: A custom CNN model is trained to classify faces based on a dataset.
- **Data Augmentation**: The model uses data augmentation to improve generalization.
- **Easy to Use**: Simple setup and clear instructions for training and testing the model.

---
## Project Structure 

facial-recognition-cnn/
│
├── dataset/                  # Folder containing the dataset
├── train_model.py            # Script to train the CNN model
├── real_time_recognition.py  # Script for real-time face recognition
├── face_recognition_cnn_improved.h5  # Trained model (generated after training)
├── README.md                 # Project documentation
└── requirements.txt          # List of dependencies

---

## Requirements

To run this project, you need the following libraries installed:

- Python 3.8 or higher
- OpenCV (`opencv-python`)
- TensorFlow (`tensorflow`)
- NumPy (`numpy`)
- scikit-learn (`scikit-learn`)

You can install the required libraries using the following command:


pip install opencv-python tensorflow numpy scikit-learn

## Dataset
The dataset should be organized as follows:

Copy
dataset/
    person_1/
        image1.jpg
        image2.jpg
        ...
    person_2/
        image1.jpg
        image2.jpg
        ...
    ...

by- Shaizan Shaikh
Mail- smdshaizan@gmail.com

