import os
import numpy as np
import pandas as pd
import cv2, imutils
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from PIL import Image, ImageChops
import imagehash

def hash_distance(im1, im2):
    """""
    Image hashes tell whether two images look nearly identical. 
    This is different from cryptographic hashing algorithms (like MD5, SHA-1) 
    where tiny changes in the image give completely different hashes. 
    In image fingerprinting, we actually want our similar inputs to have 
    similar output hashes as well.
    """
    hash0 = imagehash.average_hash(Image.fromarray(im1))
    hash1 = imagehash.average_hash(Image.fromarray(im2))
    return hash0 - hash1


def contoured_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    g = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(g, 60, 180)

    contours = cv2.findContours(edge, 
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours[0], -1, (0,0,255), thickness = 2)

    height, width, channels = image.shape
    x = height if height > width else width
    y = height if height > width else width
    square= np.zeros((height,width,3), np.uint8)
    cv2.drawContours(square, contours[0], -1, (0,0,255), thickness = 2)
    return square

def return_masks(img_path = './masks/2.jpeg', screen_path = './masks/orig/2.jpeg', n = 0.8):
    '''
    img_path = './masks/2.jpeg'
    screen_path = './masks/orig/2.jpeg'
    n = 0.8 # number of standard deviations from bottom to keep
    '''

    # find the contours in the mask
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    g = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(g, 60, 180)

    contours = cv2.findContours(edge, 
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours[0], -1, (0,0,255), thickness = 2)

    height, width, channels = image.shape
    x = height if height > width else width
    y = height if height > width else width
    square= np.zeros((height,width,3), np.uint8)
    cv2.drawContours(square, contours[0], -1, (0,0,255), thickness = 2)
    fig, ax = plt.subplots(1, figsize=(12,8))
    plt.imshow(square)

    # find on screen image and partial remove
    screen_mask = contoured_image(screen_path)
    fig, ax = plt.subplots(1, figsize=(12,8))
    plt.imshow(screen_mask)

    shots = []; coords = []; dist = [];
    for candidate_x in range(0, screen_mask.shape[1], 10):
        for candidate_y in range(0, screen_mask.shape[0], 10):
            try:
                d = hash_distance(image, screen_mask[candidate_y:candidate_y+image.shape[0], candidate_x:candidate_x+image.shape[1]])
                dist.append(d)
                shots.append(screen_mask[candidate_y:candidate_y+image.shape[0], candidate_x:candidate_x+image.shape[1]])
                coords.append([candidate_y, candidate_y+image.shape[0], candidate_x, candidate_x+image.shape[1]])
            except:
                continue

    sort = pd.DataFrame()
    sort['coords'] = coords
    sort['dist'] = dist
    sort = sort.sort_values(by='dist').reset_index(drop=True)
    lower_limit = sort['dist'][0]
    upper_limit = lower_limit+sort['dist'].std()*n
    
    patches = sort[sort['dist']<=upper_limit]['coords']
    return patches

def oneshot_inpainting(dir_path = 'visual/masks/', screen_path = './masks/orig/2.jpeg', n = 0.8):
    
    patches = [list(return_masks(dir_path+str(i), screen_path, n)) for i in os.listdir(dir_path)]
    patches = [item for sublist in patches for item in sublist]
#     patches = return_masks(img_path, screen_path, n)
    
    # return image with pixels removed
    image = cv2.imread(screen_path)
    image0 = cv2.imread(screen_path, 0)

    height, width, channels = image.shape
    x = height if height > width else width
    y = height if height > width else width
    square= np.zeros((height,width), np.uint8)
    for p in range(len(patches)):
        square[int(patches[p][0]):int(patches[p][1]), int(patches[p][2]):int(patches[p][3])] = image0[int(patches[p][0]):int(patches[p][1]), int(patches[p][2]):int(patches[p][3])]

    _, mask = cv2.threshold(square, 128, 255, cv2.THRESH_BINARY)
    mask1 = cv2.bitwise_not(mask)
    distort = cv2.bitwise_and(image, image, mask=mask1)

    output = cv2.inpaint(distort, mask, 3, cv2.INPAINT_TELEA)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    true_image = cv2.imread(screen_path)
    true_image = cv2.cvtColor(true_image,cv2.COLOR_BGR2RGB)
    ret_img = ImageChops.difference(Image.fromarray(output), Image.fromarray(true_image))
    ret_img = cv2.cvtColor(np.array(ret_img),cv2.COLOR_RGB2BGR)
    ret_img = cv2.cvtColor(np.array(ret_img),cv2.COLOR_BGR2BGRA)
    ret_img=np.where(ret_img==0, 255, ret_img) 
    return ret_img