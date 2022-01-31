# Successful oneshot detection of dark pattern interfaces with template matching
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_TEMPLATE_MATCHING_THRESHOLD = 0.5

class Template:
    """
    A class defining a template
    """
    def __init__(self, image_path, label, color, matching_threshold=DEFAULT_TEMPLATE_MATCHING_THRESHOLD):
        """
        Args:
            image_path (str): path of the template image path
            label (str): the label corresponding to the template
            color (List[int]): the color associated with the label (to plot detections)
            matching_threshold (float): the minimum similarity score to consider an object is detected by template
                matching
        """
        self.image_path = image_path
        self.label = label
        self.color = color
        self.template = cv2.imread(image_path)
        self.template_height, self.template_width = self.template.shape[:2]
        self.matching_threshold = matching_threshold


def get_iou(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2

    Arguments:
    box1 -- first box, numpy array with coordinates (ymin, xmin, ymax, xmax)
    box2 -- second box, numpy array with coordinates (ymin, xmin, ymax, xmax)
    """
    # ymin, xmin, ymax, xmax = box

    y11, x11, y21, x21 = box1
    y12, x12, y22, x22 = box2

    yi1 = max(y11, y12)
    xi1 = max(x11, x12)
    yi2 = min(y21, y22)
    xi2 = min(x21, x22)
    inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (x21 - x11) * (y21 - y11)
    box2_area = (x22 - x12) * (y22 - y12)
    union_area = box1_area + box2_area - inter_area
    # compute the IoU
    iou = inter_area / union_area
    return iou
    
def non_max_suppression(
    objects,
    non_max_suppression_threshold=0.5,
    score_key="MATCH_VALUE",
):
    """
    Filter objects overlapping with IoU over threshold by keeping only the one with maximum score.
    Args:
        objects (List[dict]): a list of objects dictionaries, with:
            {score_key} (float): the object score
            {top_left_x} (float): the top-left x-axis coordinate of the object bounding box
            {top_left_y} (float): the top-left y-axis coordinate of the object bounding box
            {bottom_right_x} (float): the bottom-right x-axis coordinate of the object bounding box
            {bottom_right_y} (float): the bottom-right y-axis coordinate of the object bounding box
        non_max_suppression_threshold (float): the minimum IoU value used to filter overlapping boxes when
            conducting non max suppression.
        score_key (str): score key in objects dicts
    Returns:
        List[dict]: the filtered list of dictionaries.
    """
    sorted_objects = sorted(objects, key=lambda obj: obj[score_key], reverse=True)
    filtered_objects = []
    for object_ in sorted_objects:
        overlap_found = False
        for filtered_object in filtered_objects:
            iou = get_iou(
                [object_['TOP_LEFT_X'], object_['TOP_LEFT_Y'], object_['BOTTOM_RIGHT_X'], object_['BOTTOM_RIGHT_Y']], 
                [filtered_object['TOP_LEFT_X'], filtered_object['TOP_LEFT_Y'], filtered_object['BOTTOM_RIGHT_X'], filtered_object['BOTTOM_RIGHT_Y']], 
            )
            if iou > non_max_suppression_threshold:
                overlap_found = True
                break
        if not overlap_found:
            filtered_objects.append(object_)
    return filtered_objects
        
    
def oneshot_templatematching(imagePath = './masks/orig/3.jpeg', mask_dir = './masks/'):
    image = cv2.imread(imagePath)

    templates = [
        Template(image_path=mask_dir+str(i), label="1", color=(0, 0, 255), matching_threshold=0.65) for i in os.listdir(mask_dir)
    ]

    detections = []
    for template in templates:
        template_matching = cv2.matchTemplate(
            template.template, image, cv2.TM_CCOEFF_NORMED
        )

        match_locations = np.where(template_matching >= template.matching_threshold)

        for (x, y) in zip(match_locations[1], match_locations[0]):
            match = {
                "TOP_LEFT_X": x,
                "TOP_LEFT_Y": y,
                "BOTTOM_RIGHT_X": x + template.template_width,
                "BOTTOM_RIGHT_Y": y + template.template_height,
                "MATCH_VALUE": template_matching[y, x],
                "LABEL": template.label,
                "COLOR": template.color
            }

            detections.append(match)

    NMS_THRESHOLD = 0.2
    detections = non_max_suppression(detections, non_max_suppression_threshold=NMS_THRESHOLD)

    image_with_detections = image.copy()
    for detection in detections:
        cv2.rectangle(
            image_with_detections,
            (detection["TOP_LEFT_X"], detection["TOP_LEFT_Y"]),
            (detection["BOTTOM_RIGHT_X"], detection["BOTTOM_RIGHT_Y"]),
            detection["COLOR"],
            2,
        )
        cv2.putText(
            image_with_detections,
            f"{detection['LABEL']} - {detection['MATCH_VALUE']}",
            (detection["TOP_LEFT_X"] + 2, detection["TOP_LEFT_Y"] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            detection["COLOR"],
            1,
            cv2.LINE_AA,
        )

    print(detections)
    return detections
    
    
    