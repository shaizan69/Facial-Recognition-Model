import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("face_recognition_cnn_improved.h5")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Define the input shape for the model
input_shape = (150, 150, 3)

# Define the label dictionary
label_dict = {
    0: "Shaizan",
    # Add more labels as needed
}

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Preprocess the frame for prediction
    resized_frame = cv2.resize(frame, (input_shape[0], input_shape[1]))  # Resize to match model input size
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    input_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

    # Predict the label
    predictions = model.predict(input_frame)
    predicted_label = np.argmax(predictions)  # Get the predicted class index
    confidence = np.max(predictions)  # Get the confidence score

    # Get the person's name from the label dictionary
    person_name = label_dict.get(predicted_label, "Unknown")

    # Display the name and confidence on the frame
    cv2.putText(frame, f"{person_name} ({confidence:.2f})", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-Time Face Recognition', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()