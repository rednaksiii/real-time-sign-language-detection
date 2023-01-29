import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
img_size = 400
counter = 0
folder = "Data/D"
imgWhite = None

while True:
    success, img = capture.read()
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

        else:
            const_num = img_size / width
            height_calculated = math.ceil(const_num * height)
            img_resized = cv2.resize(imgCrop, (img_size, height_calculated))

            img_resized_shape = img_resized.shape
            height_gap = math.ceil((img_size - height_calculated)/2)
            imgWhite[height_gap:height_calculated+height_gap, :] = img_resized

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

