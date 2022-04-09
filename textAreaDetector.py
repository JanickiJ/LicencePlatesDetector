from imutils.object_detection import non_max_suppression
import cv2
import numpy as np
from config import MIN_CONFIDENCE, NETWORK_MODEL_PATH, NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT

net = None

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
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < MIN_CONFIDENCE:
				continue
			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

def loadNet():
    global net
    net = cv2.dnn.readNet(NETWORK_MODEL_PATH)

def textAreas(image):
    global net
    orig = image.copy()
    (H, W) = image.shape[:2]
    width_ratio = W / float(NETWORK_IMAGE_WIDTH)
    height_ratio = H / float(NETWORK_IMAGE_HEIGHT)

    # resize the image and grab the new image dimensions
    network_image = cv2.resize(image, (NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT))
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    blob = cv2.dnn.blobFromImage(network_image, 1.0, (NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT),
                            (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net.setInput(blob)
    scores, geometry = net.forward(layerNames)
    rects, confidences = decode_predictions(scores=scores, geometry=geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    result = []
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * width_ratio)
        startY = int(startY * height_ratio)
        endX = int(endX * width_ratio)
        endY = int(endY * height_ratio)
        result.append((startX, startY, endX, endY))
    return result