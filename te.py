import re

import pytesseract
import numpy as np
import imutils
import cv2


def work(debug=False):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
        edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
        res = frame.copy()
        try:
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = cv2.bitwise_and(frame, frame, mask=mask)
            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
            if debug:
                res = cv2.rectangle(res, (y2, x2), (y1, x1), (0, 255, 0), 3)
            text = pytesseract.image_to_string(cropped_image, config='--psm 7')
            parsed_text = re.sub(r'[^A-Z0-9]', '', text)
            if len(parsed_text)>5:
                print(parsed_text)
        except:
            pass
        if debug:
            cv2.imshow('frame', res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


work()
