from time import sleep
from config import Config as config
from imutils.object_detection import non_max_suppression
import cv2
import numpy as np
from logger import console_logger


def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scores_data[x] < config.MIN_CONFIDENCE:
                continue
            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height
            # of the bounding box
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]
            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            end_x = int(offsetX + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offsetY - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)
            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])
    # return a tuple of the bounding boxes and associated confidences
    return rects, confidences


def camera():
    # place an object with text in front of webcam and check results
    # press 'q' to stop the program
    console_logger.info("loading EAST text detector...")
    net = cv2.dnn.readNet(config.NETWORK_MODEL_PATH)
    console_logger.info("start capturing webcam...")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, image = cap.read()
        orig = image.copy()
        (H, W) = image.shape[:2]
        width_ratio = W / float(config.NETWORK_IMAGE_WIDTH)
        height_ratio = H / float(config.NETWORK_IMAGE_HEIGHT)

        # resize the image and grab the new image dimensions
        network_image = cv2.resize(image, (config.NETWORK_IMAGE_WIDTH, config.NETWORK_IMAGE_HEIGHT))
        layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        console_logger.info("converting image to blob...")
        blob = cv2.dnn.blobFromImage(network_image, 1.0, (config.NETWORK_IMAGE_WIDTH, config.NETWORK_IMAGE_HEIGHT),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)

        net.setInput(blob)
        scores, geometry = net.forward(layer_names)
        rects, confidences = decode_predictions(scores=scores, geometry=geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        # loop over the bounding boxes
        for (start_x, start_y, end_x, end_y) in boxes:
            # scale the bounding box coordinates based on saved ratios
            start_x = int(start_x * width_ratio)
            start_y = int(start_y * height_ratio)
            end_x = int(end_x * width_ratio)
            end_y = int(end_y * height_ratio)
            # draw green rect
            cv2.rectangle(orig, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        cv2.imshow('Webcam', orig)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        sleep(0.1)


if __name__ == '__main__':
    camera()
