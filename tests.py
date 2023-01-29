import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")


offset = 20
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z"]
img_size = 400
counter = 0
folder = "Data/D"
imgWhite = None

while True:
    success, img = capture.read()
    image_output = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, width, height = hand["bbox"]
        imgCrop = img[y-offset:y+height+offset, x-offset:x+width+offset]

        imgWhite = np.ones((img_size, img_size, 3), np.uint8)*255

        imgCropShape = imgCrop.shape
        aspect_ratio = height / width

        if aspect_ratio > 1:
            const_num = img_size / height
            width_calculated = math.ceil(const_num * width)
            img_resized = cv2.resize(imgCrop, (width_calculated, img_size))

            img_resized_shape = img_resized.shape

            width_gap = math.ceil((img_size - width_calculated)/2)

            imgWhite[:, width_gap:width_calculated + width_gap] = img_resized
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)


        else:
            const_num = img_size / width
            height_calculated = math.ceil(const_num * height)
            img_resized = cv2.resize(imgCrop, (img_size, height_calculated))

            img_resized_shape = img_resized.shape
            height_gap = math.ceil((img_size - height_calculated)/2)
            imgWhite[height_gap:height_calculated+height_gap, :] = img_resized
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        cv2.rectangle(image_output, (x-offset, y-offset-50), (x-offset+100, y-offset), (255, 0, 255), cv2.FILLED)

        cv2.putText(image_output, labels[index], (x, y-25), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(image_output, (x-offset, y-offset), (x+width+offset, y+height+offset), (255, 0, 255), 2)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", image_output)
    key = cv2.waitKey(1)
