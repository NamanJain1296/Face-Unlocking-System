import cv2
import os

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

face_id = input('\nEnter User ID and Press <return> ==> ')
save_directory = r'C:\Users\naman\PycharmProjects\Face Unlocking System\dataset'
print("\n*** Initializing face capture. Look at the camera and wait ... ****")

count = 0

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

        cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif count >= 30:
        break

cap.release()
cv2.destroyAllWindows()
