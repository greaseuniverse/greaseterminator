"""
Flow:
1. use region clustering to identify main regions
2. identify useful segments of the image -> cut up each image into sub images
3. cross-correlate between the 2 image segments, and if there is image segment that is both (i) significant portion of the screen & (ii) high image similarity, then it must have been a scrolled instance
"""

import numpy as np
import pandas as pd
import cv2, imutils
import matplotlib.pyplot as plt
from PIL import Image
import imagehash

##################### GENERAL OPACITY ADJUSTMENT ########################
def opacity(img, level):
    """
    100% transparent: 0
    100% opaque: 255
    """
    img[:, :, 3] = level
    return img

##################### DARKENING OF SCREEN ########################
def darken(square, trigger, dark_state):
    if trigger == True:
        dark_state['current_level'] += dark_state['iterator']
        print(min(max(dark_state['min_level'], dark_state['current_level']), dark_state['max_level']))
        square = opacity(square, min(max(dark_state['min_level'], dark_state['current_level']), dark_state['max_level']))
        return square, dark_state
    if trigger == False:
        return square, dark_state

##################### SIMIILARITY SCORING ########################    
# Function to fill all the bounding box
def fill_rects(image, stats):

    for i,stat in enumerate(stats):
        if i > 0:
            p1 = (stat[0],stat[1])
            p2 = (stat[0] + stat[2],stat[1] + stat[3])
            cv2.rectangle(image,p1,p2,255,-1)

def relevant_regions(img_path = '../repo/docs/test_screen.png', img_prev_path = '../repo/docs/test_screen_prev.png'):
    """
    Target / result diff image is img2, i.e. img_path
    """ 
    # Load image file
    img1 = cv2.imread(img_prev_path,0)
    img2 = cv2.imread(img_path,0)

    # Subtract the 2 image to get the difference region
    img3 = cv2.subtract(img1,img2)

    # Make it smaller to speed up everything and easier to cluster
    small_img = cv2.resize(img3,(0,0),fx = 0.25, fy = 0.25)


    # Morphological close process to cluster nearby objects
    fat_img = cv2.dilate(small_img, None,iterations = 3)
    fat_img = cv2.erode(fat_img, None,iterations = 3)

    fat_img = cv2.dilate(fat_img, None,iterations = 3)
    fat_img = cv2.erode(fat_img, None,iterations = 3)

    # Threshold strong signals
    _, bin_img = cv2.threshold(fat_img,20,255,cv2.THRESH_BINARY)

    # Analyse connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)

    # Cluster all the intersected bounding box together
    rsmall, csmall = np.shape(small_img)
    new_img1 = np.zeros((rsmall, csmall), dtype=np.uint8)

    fill_rects(new_img1,stats)


    # Analyse New connected components to get final regions
    num_labels_new, labels_new, stats_new, centroids_new = cv2.connectedComponentsWithStats(new_img1)


    labels_disp = np.uint8(200*labels/np.max(labels)) + 50
    labels_disp2 = np.uint8(200*labels_new/np.max(labels_new)) + 50

    rescaled_img = cv2.resize(labels_disp2,(0,0),fx = 1/0.25, fy = 1/0.25)

    return rescaled_img

def height_ops(img_path, img_prev_path):
    diff = relevant_regions(img_path, img_prev_path)
    rows_mean = [i.mean() for i in diff]
    unique, counts = np.unique(rows_mean, return_counts=True)
    y_sets = []
    tmp=[]
    for i in range(1, len(rows_mean)):
        if len(tmp) < 1:
            if float(rows_mean[i-1]) != unique[0]:
                tmp.append(i-1)

        if len(tmp) == 1:
            if float(rows_mean[i-1]) != unique[0]:
                if float(rows_mean[i]) == unique[0]:
                    tmp.append(i-1)
                    y_sets.append(tmp)
                    tmp=[]
    return y_sets

def comparable_heights(img_path = '../repo/docs/test_screen.png', img_prev_path = '../repo/docs/test_screen_prev.png'):
    y_sets_1 = height_ops(img_path, img_prev_path)
    y_sets_2 = height_ops(img_prev_path, img_path)
    return y_sets_1, y_sets_2


def ret_similar(img_path = '../repo/docs/test_screen.png', img_prev_path = '../repo/docs/test_screen_prev.png', size_threshold = 0.4):

    y_sets_1, y_sets_2 = comparable_heights(img_path, img_prev_path)

    img = Image.open(img_path)
    img_prev = Image.open(img_prev_path)

    img_set = []
    img_prev_set = []

    for y_set in [y_sets_1, y_sets_2]:
        for j in range(len(y_set)):
            img_set.append(img.crop((0, y_set[j][0], img.size[1], y_set[j][1]))) 
            img_prev_set.append(img_prev.crop((0, y_set[j][0], img_prev.size[1], y_set[j][1]))) 

    correl_matrix = []
    for img_item in img_set:
        for img_prev_item in img_prev_set:
            hash0 = imagehash.average_hash(img_item) 
            hash1 = imagehash.average_hash(img_prev_item) 
            diff = hash0 - hash1
            correl_matrix.append([img_item, img_prev_item, diff])
        
        
    # similarity threshold should be (i) 0 avg hash diff & (ii) user-defined threshold on min size wrt whole img
    similar = False
    for k in range(len(correl_matrix)):
        candidate = correl_matrix[k]
        if candidate[2] == 0:
            if float(candidate[0].size[1])/img.size[0] >= size_threshold:
                if float(candidate[1].size[1])/img.size[0] >= size_threshold:
                    similar = True
    return similar
