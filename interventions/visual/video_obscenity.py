import cv2
import numpy as np
from nudenet import NudeDetector

def image_censor(image_in):
    detector = NudeDetector()
    loaded_image = cv2.imread(image_in)
    mask= np.zeros((loaded_image.shape[0], loaded_image.shape[1], 3), np.uint8)
    mask=np.where(mask==0, 255, mask) 
    coords = [i['box'] for i in detector.detect(image_in)]
    print("Obs: ", coords)
    for box in coords:
        cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), -1)
        cv2.GaussianBlur(mask[box[1]:box[3], box[0]:box[2]], (23, 23), 30)
    return mask, coords