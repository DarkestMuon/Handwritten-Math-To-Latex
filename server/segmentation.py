import cv2
import numpy as np

def segmentation(image_path):

    # loading image into cv2 and grayscaling
    image = cv2.imread(image_path)
    image_grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image_grayscaled, 0, 150)
    blurred = cv2.blur(edges, (8,8), 0)

    # finding contours within the image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # areas we want the contour to be inbetween
    minimum_area, maximum_area = 200, 60000

    # make a copy of the image to draw the detections on
    image_copy = image.copy()
    boundaries = []

    for iter, contour in enumerate(contours):

        # if the area is too large or small, we want to ignore it
        area = cv2.contourArea(contour)

        if area > maximum_area or area < minimum_area:
            continue
    
        # detect contours
        (x, y, w, h) = cv2.boundingRect(contour)

        # appends only if contour is not inside another contour
        if hierarchy[0, iter, 3] == -1:
            boundaries.append((x, y, w, h))

    # sort the contours from left to right (based on x-coordinate)
    boundaries.sort(key= lambda x: x[0])

    # save region of interests (ROI's) from left to right
    for iter2, boundary in enumerate(boundaries):

        x = boundary[0]
        y = boundary[1]
        w = boundary[2]
        h = boundary[3]

        # draws rectangle around the detection of the symbol
        # not neccessary unless you want to see the image being drawn on (line 57-58)

        # cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # take ROI (region of interest)
        roi = image[y:y+h, x:x+w]

        # shows the object detections - USED IN DEMO
        # cv2.imshow('', image_copy)
        # cv2.waitKey(0)

        # save the ROI as an image to rois folder (later accessed by classifier.py)
        cv2.imwrite('../rois/' + str(iter2+1) + '.png', roi)