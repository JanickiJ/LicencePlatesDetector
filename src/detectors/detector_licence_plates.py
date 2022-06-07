import re
import pytesseract
import numpy as np
import imutils
import cv2


def set_windows_tess_path():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class DetectorLicencePlates:
    def __init__(self, debug=False):
        self.debug = debug

    def run(self, frame):
        parsed_text, debug_image = None, None
        gray_image = self.gray_image(frame)
        edged_image = self.edged_image(gray_image)
        contours = self.get_contours(edged_image)
        cropped_image, debug_image = self.crop_plate(gray_image, frame, contours)
        if cropped_image is not None:
            text_from_image = self.image_to_text(cropped_image)
            parsed_text = self.parse_text(text_from_image)
        return parsed_text, debug_image

    def crop_plate(self, gray_image, frame, location):
        cropped_image = None
        debug_image = None
        try:
            mask = np.zeros(gray_image.shape, np.uint8)
            cv2.drawContours(mask, [location], 0, 255, -1)
            cv2.bitwise_and(frame, frame, mask=mask)
            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray_image[x1:x2 + 1, y1:y2 + 1]
            if self.debug:
                debug_image = cv2.rectangle(frame.copy(), (y2, x2), (y1, x1), (0, 255, 0), 3)
        except Exception as ex:
            pass
        return cropped_image, debug_image

    @staticmethod
    def gray_image(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def edged_image(gray_image):
        bfilter = cv2.bilateralFilter(gray_image, 11, 17, 17)  # Noise reduction
        edged = cv2.Canny(bfilter, 30, 200)
        return edged

    @staticmethod
    def get_contours(parsed_frame):
        keypoints = cv2.findContours(parsed_frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
        return location

    @staticmethod
    def image_to_text(cropped_image):
        return pytesseract.image_to_string(cropped_image, config='--psm 7')

    @staticmethod
    def parse_text(text):
        parsed_text = re.sub(r'[^A-Z0-9]', '', text)
        if len(parsed_text) > 5:
            return parsed_text
