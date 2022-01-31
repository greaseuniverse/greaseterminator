import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sched, time

from text.text_filter import *
from visual.inpainting import *
from visual.template_matching import *
from visual.darken import *
from visual.video_obscenity import *
from PIL import Image


def run(image_in = '../relay/img/screen.png', image_prev = '../relay/img/screen_prev.png', image_tmpout = '../relay/img/screen.png', image_out = '../relay/img/square_img.png'):
    
    dark_state = {
                    'current_level' : 0,
                    'min_level' : 0,
                    'max_level' : 255,
                    'iterator' : 50,
                    'similarity_list' : [],
                    'similarity_count' : 10,
                 }
    trigger = False
    
    while True:

        # settings
        hate_speech_text_censor = True
        oneshot_bbox = True
        visual = False
        scroll_darken = True
        video_obscenity_censor = False

        # Screen capture
        os.system("adb exec-out screencap -p > "+str(image_in))
        time.sleep(100/1000)
        # remove status bar & nav bar
        img = cv2.cvtColor(cv2.imread(image_in, cv2.IMREAD_COLOR), cv2.COLOR_RGB2RGBA)
#         img_prev = cv2.cvtColor(cv2.imread(image_in, cv2.IMREAD_COLOR), cv2.COLOR_RGB2RGBA)
        height, width, channels = img.shape
        crop_img = img[int(height*0.04):int(height*0.925), ]
        cv2.imwrite(image_tmpout, crop_img)

        # transparency
        img = cv2.cvtColor(cv2.imread(image_in, cv2.IMREAD_COLOR), cv2.COLOR_RGB2RGBA)
        height, width, channels = img.shape
        img_prev = img.copy()
        img = opacity(img, 255)

        # mask layout
        mask= np.zeros((height, width, 3), np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2RGBA)
        mask = opacity(mask, 0)
        if hate_speech_text_censor == True:
            band_mask, reshaped_boxes = content_filtering(image_in)
            band_mask = cv2.cvtColor(band_mask, cv2.COLOR_RGB2RGBA)
            band_mask = opacity(band_mask, 128)
            print(reshaped_boxes)
        if visual == True:
            inpainted_mask = oneshot_inpainting('visual/masks/', image_in, 0.8)
            inpainted_mask = cv2.cvtColor(inpainted_mask, cv2.COLOR_RGB2RGBA)
            inpainted_mask = opacity(inpainted_mask, 128)
            

        # generates square image for Android
        x = height if height > width else width
        y = height if height > width else width
        square= np.zeros((height,height,3), np.uint8)
        square = cv2.cvtColor(square, cv2.COLOR_RGB2RGBA)
        square = opacity(square, 0)
        if hate_speech_text_censor == True:
            square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = band_mask
            square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = cv2.GaussianBlur(square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)], (23, 23), 30)
#             for obs_box in reshaped_boxes:
#                 square[int((y-height)/2+obs_box[1]):int((y-height)/2+obs_box[3]), int((x-width)/2+obs_box[0]):int((x-width)/2+obs_box[2]), 3] = 255
        if visual == True:
            square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = inpainted_mask
        if oneshot_bbox == True:
            detections = oneshot_templatematching(image_in, mask_dir = './visual/masks/')
            for detection in detections:
                margin_control = 0.2
                cv2.rectangle(
                    square,
                    (int(detection["TOP_LEFT_X"]+height/4-height*margin_control), int(detection["TOP_LEFT_Y"]-height*margin_control)),
                    (int(detection["BOTTOM_RIGHT_X"]+height/4+height*margin_control), int(detection["BOTTOM_RIGHT_Y"]+height*margin_control)),
                    detection["COLOR"],
                    2,
                )  
                square[detection["TOP_LEFT_Y"]:detection["BOTTOM_RIGHT_Y"], int(detection["TOP_LEFT_X"]+height/4):int(detection["BOTTOM_RIGHT_X"]+height/4), 3] = 255

        if scroll_darken == True:
            if os.path.exists(image_prev) == False:
                cv2.imwrite(image_prev, img_prev)
            if os.path.exists(image_prev) == True:
                # Insert trigger conditions here -- time / scroll
                print(dark_state['similarity_list'])
                similarity = ret_similar(image_in, image_prev, size_threshold = 0.25)
                if len(dark_state['similarity_list']) < 10:
                    dark_state['similarity_list'].append(similarity)
                if len(dark_state['similarity_list']) == 10:
                    dark_state['similarity_list'] = dark_state['similarity_list'][1:]
                    dark_state['similarity_list'].append(similarity)
                    if set(dark_state['similarity_list']) == set([True]):
                        trigger = True
#                     if set(dark_state['similarity_list']) != set([True]):
#                         dark_state['current_level'] = 0
                square, dark_state = darken(square, trigger, dark_state)
                cv2.imwrite(image_prev, img_prev)
                trigger = False

        if video_obscenity_censor == True:
            obscenity_mask, coords = image_censor(image_in)
            obscenity_mask = cv2.cvtColor(obscenity_mask, cv2.COLOR_RGB2RGBA)
            obscenity_mask = opacity(obscenity_mask, 128)
            square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = obscenity_mask
            for obs_box in coords:
                square[int((y-height)/2+obs_box[1]):int((y-height)/2+obs_box[3]), int((x-width)/2+obs_box[0]):int((x-width)/2+obs_box[2]), 3] = 255
#                 square[obs_box[1]:obs_box[3], obs_box[0]:obs_box[2], 3] = 255
            cv2.GaussianBlur(square, (23, 23), 11)

        cv2.imwrite(image_out, square)
        

# main
run(image_in = '../relay/img/screen.png', image_out = '../relay/img/square_img.png')