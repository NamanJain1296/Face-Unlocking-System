import cv2
import os
import numpy as np

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eyeCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

face_id = input('\nEnter User ID and Press <return> ==> ')
save_directory = r'C:\Users\naman\PycharmProjects\Face Unlocking System\dataset'
print("\n*** Initializing face capture. Look at the camera and wait ... ****")

count = 0

def eye_aspect_ratio(eye):
    vertical_dist1 = np.linalg.norm(eye[1] - eye[5])
    vertical_dist2 = np.linalg.norm(eye[2] - eye[4])

    horizontal_dist = np.linalg.norm(eye[0] - eye[3])
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        image_path = os.path.join(save_directory, f"User_{face_id}_{count}.jpg")
        cv2.imwrite(image_path, gray[y:y + h, x:x + w])
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=10,
            minSize=(5, 5),
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        left_eye_roi = gray[y:y+h, x:x+w][:, :w//2]
        right_eye_roi = gray[y:y+h, x:x+w][:, w//2]

        left_eye_ear = eye_aspect_ratio(left_eye_roi)
        right_eye_ear = eye_aspect_ratio(right_eye_roi)

        blink_ear_threshold = 0.2
        if left_eye_ear < blink_ear_threshold and right_eye_ear < blink_ear_threshold:
            print("Blink detected!")

        cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif count >= 30:
        break

cap.release()
cv2.destroyAllWindows()
