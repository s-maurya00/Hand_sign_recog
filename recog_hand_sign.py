import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

OFFSET = 20
WINDOW_FRAME_SIZE_FOR_HAND = 300
SAVE_LOCATION = "Data/error"


labels = ["A", "B", "C"]

while True:
    success, frame = cap.read()

    output_frame = frame.copy() 

    hands= detector.findHands(frame, draw = False)

    if hands:

        hand = hands[0]
        x, y, width, height = hand['bbox']
        cropped_to_hand_frame = frame[y - OFFSET: y + height + OFFSET, x - OFFSET: x + width + OFFSET]

        # Empty window frame to show cropped hand
        white_frame = np.ones((WINDOW_FRAME_SIZE_FOR_HAND, WINDOW_FRAME_SIZE_FOR_HAND, 3), np.uint8) * 255


        # Stretching the image
        aspect_ratio = height / width
        try:
            # set height to WINDOW_FRAME_SIZE and scale width as per scaling factor
            if aspect_ratio > 1:
                frame_scale_factor = WINDOW_FRAME_SIZE_FOR_HAND / height

                width_scaled = math.ceil(frame_scale_factor * width)

                frame_hand_resized = cv2.resize(cropped_to_hand_frame, (width_scaled, WINDOW_FRAME_SIZE_FOR_HAND))

                width_gap = math.ceil((WINDOW_FRAME_SIZE_FOR_HAND - width_scaled) / 2)

                white_frame[ : , width_gap : (width_gap + width_scaled)] = frame_hand_resized

                prediction, index = classifier.getPrediction(white_frame, draw = False)
                cv2.putText(output_frame, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

                print(prediction, index)


            # set width to WINDOW_FRAME_SIZE and scale height as per scaling factor
            else:
                frame_scale_factor = WINDOW_FRAME_SIZE_FOR_HAND / width

                height_scaled = math.ceil(frame_scale_factor * height)

                frame_hand_resized = cv2.resize(cropped_to_hand_frame, (WINDOW_FRAME_SIZE_FOR_HAND, height_scaled))
                
                height_gap = math.ceil((WINDOW_FRAME_SIZE_FOR_HAND - height_scaled) / 2)

                white_frame[height_gap : (height_gap + height_scaled), : ] = frame_hand_resized

                prediction, index = classifier.getPrediction(white_frame, draw = False)
                cv2.putText(output_frame, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)




        except:
            print("Hand out of frame")

        if ((cropped_to_hand_frame is not None) and (cropped_to_hand_frame.size != 0)):

            try:
                cv2.imshow("White Frame", white_frame)
            except:
                print("Hand exceeded the window size")

    cv2.imshow("Output Frame", output_frame)

    cv2.waitKey(1)