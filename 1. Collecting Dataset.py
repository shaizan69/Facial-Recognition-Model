import cv2
import os

# Create dataset folder if it doesn't exist
if not os.path.exists("Dataset"):
    os.makedirs("Dataset")

# Input name of the person
person_name = input("Enter the name of the person: ")
person_folder = os.path.join("Dataset", person_name)

if not os.path.exists(person_folder):
    os.makedirs(person_folder)

# Initialize webcam
video_capture = cv2.VideoCapture(0)
count = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Display the frame
    cv2.imshow('Capturing Images', frame)

    # Save the image when 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        image_path = os.path.join(person_folder, f"{person_name}_{count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved {image_path}")
        count += 1

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()

