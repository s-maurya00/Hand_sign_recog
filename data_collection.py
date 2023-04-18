import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math, time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

OFFSET = 20
WINDOW_FRAME_SIZE_FOR_HAND = 300
COUNTER = 0
SAVE_LOCATION = "Data/Z"


while True:
    success, frame = cap.read()

    hands, frame = detector.findHands(frame)

    if hands:

        hand = hands[0]
        x, y, w, h = hand['bbox']
        cropped_to_hand_frame = frame[y - OFFSET: y + h + OFFSET, x - OFFSET: x + w + OFFSET]

        # Empty window frame to show cropped hand
        white_frame = np.ones((WINDOW_FRAME_SIZE_FOR_HAND, WINDOW_FRAME_SIZE_FOR_HAND, 3), np.uint8) * 255


        # Stretching the image
        aspect_ratio = h / w
        try:
            # set height to WINDOW_FRAME_SIZE and scale width as per scaling factor
            if aspect_ratio > 1:
                frame_scale_factor = WINDOW_FRAME_SIZE_FOR_HAND / h

                w_scaled = math.ceil(frame_scale_factor * w)

                frame_hand_resized = cv2.resize(cropped_to_hand_frame, (w_scaled, WINDOW_FRAME_SIZE_FOR_HAND))

                w_scaled_gap = math.ceil((WINDOW_FRAME_SIZE_FOR_HAND - w_scaled) / 2)

                # white_frame[0: resized_frame.shape[0], 0: resized_frame.shape[1]] = resized_frame
                white_frame[ : , w_scaled_gap: w_scaled_gap + w_scaled] = frame_hand_resized


            # set width to WINDOW_FRAME_SIZE and scale height as per scaling factor
            else:
                frame_scale_factor = WINDOW_FRAME_SIZE_FOR_HAND / w

                h_scaled = math.ceil(frame_scale_factor * h)

                frame_hand_resized = cv2.resize(cropped_to_hand_frame, (WINDOW_FRAME_SIZE_FOR_HAND, h_scaled))
                
                h_gap = math.ceil((WINDOW_FRAME_SIZE_FOR_HAND - h_scaled) / 2)

                # white_frame[0: resized_frame.shape[0], 0: resized_frame.shape[1]] = resized_frame
                white_frame[h_gap: h_gap + h_scaled, : ] = frame_hand_resized

        except:
            print("Hand out of frame")

        if ((cropped_to_hand_frame is not None) and (cropped_to_hand_frame.size != 0)):
            # cv2.imshow("Cropped Frame", cropped_frame)
            try:
                cv2.imshow("White Frame", white_frame)
            except:
                print("Hand exceeded the window size")

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if(key == ord("s")):
        COUNTER += 1
        cv2.imwrite(f"{SAVE_LOCATION}/Image_{time.time()}.jpg", white_frame)
        print("Image: ", COUNTER)
