import cv2
import numpy as np
from .ocr_redaction import OCR
from .speech_filter import hate_speech_detection
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def content_filtering(image = '../relay/img/screen.png'):
    east = 'text/data/frozen_east_text_detection.pb' # path to input EAST text detector
    extractor = OCR(east)

    # Inputs
    confidence = 0.5 # minimum probability required to inspect a region
    width = 320 # resized image width (should be multiple of 32)
    height = 320 # resized image height (should be multiple of 32)
    display = False # Display bounding boxes
    numbers = False # Detect only numbers
    percentage = 2.0 # Expand/shrink detected bound box
    min_boxes = 1 # minimum number of detected boxes to return
    max_iterations = 20 # max number of iterations finding min_boxes

    ocr_data, size = extractor.get_image_text(
                                       image,
                                       width,
                                       height,
                                       display,
                                       numbers,
                                       confidence,
                                       percentage,
                                       min_boxes,
                                       max_iterations)

    texts = ocr_data['text'] # ocr_data.keys()
    left = ocr_data['left']
    top = ocr_data['top']
    width_ocr = ocr_data['width']
    height_ocr = ocr_data['height']
    band = True
    print(texts)
    
    loaded_image = cv2.imread(image)
    mask= np.zeros((loaded_image.shape[0], loaded_image.shape[1], 3), np.uint8)
    mask=np.where(mask==0, 255, mask) 
    
    orig_boxes = [[left[i], top[i], left[i]+width_ocr[i], top[i]+height_ocr[i]] for i in range(len(left))]
    box_proportions = [[left[i]/size[0], top[i]/size[1], (left[i]+width_ocr[i])/size[0], (top[i]+height_ocr[i])/size[1]] for i in range(len(left))]
    reshaped_boxes = [[int(box[0]*loaded_image.shape[1]), int(box[1]*loaded_image.shape[0]), int(box[2]*loaded_image.shape[1]), int(box[3]*loaded_image.shape[0])] for box in box_proportions]

    # words to filter out; can replace with topical model / fake news detection model
#     blacklist = [s.lower() for s in ['COVID-19', 'Trump', 'Queen']]
    blacklist = [s.lower() for s in hate_speech_detection(texts)]
    
    trigger = False; counter = 0;
    for start_X, start_Y, end_X, end_Y in reshaped_boxes:
        if texts[counter].lower() in blacklist:
            trigger = True
            # color the word only
            if band == False:
                cv2.rectangle(mask, (start_X, start_Y), (end_X, end_Y), (0, 255, 0), 2)
            if band == True:
                cv2.rectangle(mask, (0, start_Y), (loaded_image.shape[0], end_Y), (0, 0, 0), -1)
            trigger = False
        counter+=1
    return mask, reshaped_boxes
