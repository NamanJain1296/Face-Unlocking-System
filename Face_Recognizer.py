import cv2
import dlib
import numpy as np
import time

# Paths to the shape predictor and trained model
predictor_path = r"C:\Users\naman\PycharmProjects\Face Unlocking System\shape_predictor_68_face_landmarks.dat"
trainer_path = r'C:\Users\naman\PycharmProjects\Face Unlocking System\trainer\trainer.yml'

# Initialize the dlib shape predictor
predictor = dlib.shape_predictor(predictor_path)

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

# Load the face cascade classifier
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

# List of names corresponding to user IDs
names = ['None', 'Naman', 'Dhoop', 'Saksham', 'Parth', 'Sudhanshu', 'Sahil', 'Anubhav']

# Initialize the webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Minimum size for face detection
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

def unlock_system(user):
    print(f"Welcome, {user}! Unlocking System activated.")

def unknown(user):
    print(f"You are not allowed to access the security system")

confidence_threshold = 30

# Variable to track if a user has been recognized
recognized = False

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Predict the user ID and confidence
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        confidence = int(confidence)

        if confidence < 100:
            user = names[id]
            confidence_percent = round(100 - confidence)

            # Display user ID and confidence
            if confidence_percent > confidence_threshold:
                unlock_system(user)
                recognized = True
                break  # Exit the for-loop to start the 3-second timer
            else:
                unknown(user)
        else:
            user = "unknown"
            unknown(user)

        cv2.putText(img, str(user), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    # Show the frame with detections
    cv2.imshow('camera', img)

    # Exit the program when the Escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Exit the program after 3 seconds if recognized
    if recognized:
        time.sleep(3)
        break

print("\n *** Exiting Program and Cleaning up stuff ***")
cam.release()
cv2.destroyAllWindows()
