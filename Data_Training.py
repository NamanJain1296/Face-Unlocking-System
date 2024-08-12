import cv2
import os
import numpy as np
from PIL import Image

data_path = r'C:\Users\naman\PycharmProjects\Face Unlocking System\dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def getImagesAndLabels(path):
    imagesPaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagesPaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        filename = os.path.splitext(os.path.basename(imagePath))[0]
        user_id = int(filename.split("_")[1])
        faces = faceCascade.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(user_id)

    return faceSamples, ids

print("\n*** Training faces. It will take a few seconds. Wait ... ***")

faces, ids = getImagesAndLabels(data_path)
trainer_path = r'C:\Users\naman\PycharmProjects\Face Unlocking System\trainer\trainer.yml'
recognizer.train(faces, np.array(ids))
recognizer.write(trainer_path)
print("\n *** {0} faces trained. Exiting Program ***".format(len(np.unique(ids))))
