import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:\\Users\\naman\\PycharmProjects\\Face Unlocking System\\trainer\\trainer.yml')

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
font = cv2.FONT_HERSHEY_SIMPLEX

names = ['None', 'Naman', 'Dhoop', 'Saksham', 'Parth', 'Sudhanshu', 'Sahil', 'Anubhav']

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

def eye_aspect_ratio(eye):
    vertical_dist1 = np.linalg.norm(eye[1] - eye[5])
    vertical_dist2 = np.linalg.norm(eye[2] - eye[4])

    horizontal_dist = np.linalg.norm(eye[0] - eye[3])
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

def unlock_system(user):
    print(f"Welcome, {user}! Unlocking System activated.")

def unknown(user):
    print(f"You are not allowed to access the security system")

confidence_threshold = 30

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        confidence = int(confidence)

        print(f"Predicted ID: {id}, Confidence: {confidence}")

        if confidence < 100:
            user = names[id]
            confidence_percent = round(100 - confidence)
            confidence = f"  {confidence_percent}%"
            if confidence_percent > confidence_threshold:
                unlock_system(user)
            else:
                unknown(user)
        else:
            user = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(
            img,
            str(user),
            (x + 5, y - 5),
            font,
            1,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img,
            str(confidence),
            (x + 5, y + h - 5),
            font,
            1,
            (255, 255, 0),
            1
        )

        left_eye_roi = gray[y:y + h, x:x + w][:, :w // 2]
        right_eye_roi = gray[y:y + h, x:x + w][:, w // 2:]

        left_eye_ear = eye_aspect_ratio(left_eye_roi)
        right_eye_ear = eye_aspect_ratio(right_eye_roi)

        blink_ear_threshold = 0.2
        if left_eye_ear < blink_ear_threshold and right_eye_ear < blink_ear_threshold:
            print("Blink detected!")

    cv2.imshow('camera', img)

    if cv2.waitKey(10) == 27 or len(faces) > 0:
        break

print("\n *** Exiting Program and Cleaning up stuff ***")
cam.release()
cv2.destroyAllWindows()
