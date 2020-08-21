# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 19:51:28 2020

@author: Abhijeet
"""


from .social_distancing_configuration import MIN_CONF
from .social_distancing_configuration import NMS_THRESH

import numpy as np
import cv2


#Take Frames from SD
#Pre Process
#Frames, Give back to the model
#Get Outputs from model
#compared - Only Persons returned
#Non Maxima Suppression
#Centroid, B Box Cord, Confidence

def detect_people(frame, net, ln, personIdx=0):
    (H, W) = frame.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            #confidence, x, y, h, w, pr(1), pr(2), pr(3)

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID == personIdx and confidence > MIN_CONF:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width,height)=box.astype("int")
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes,confidences,MIN_CONF,NMS_THRESH)

    if len(idxs) > 0:
        for i in idxs.flatten():
            #extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h)= (boxes[i][2], boxes[i][3])

    		# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # return the list of results
    return results




