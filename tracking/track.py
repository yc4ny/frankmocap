import cv2
import numpy as np
import os
import subprocess 

def check_bbox_consistency(curr_bbox, prev_bbox):
    x_diff = abs(curr_bbox[0] - prev_bbox[0])
    y_diff = abs(curr_bbox[1] - prev_bbox[1])
    return x_diff > 30 or y_diff > 30


def check_negative_numbers(hand_bbox_list):
    for box_dict in hand_bbox_list:
        for key, value in box_dict.items():
            if np.any(value < 0) or np.any(value >3820):
                return True
    if hand_bbox_list[0]['left_hand'][2] + hand_bbox_list[0]['left_hand'][0] > 3900:
        return True

    if hand_bbox_list[0]['left_hand'][3] + hand_bbox_list[0]['left_hand'][1] > 2200:
        return True

    if hand_bbox_list[0]['right_hand'][2] + hand_bbox_list[0]['right_hand'][0] > 3900:
        return True

    if hand_bbox_list[0]['right_hand'][3] + hand_bbox_list[0]['right_hand'][1] > 2200:
        return True
    
    if hand_bbox_list[0]['left_hand'][0] < 1200:
        return True
    
    if hand_bbox_list[0]['left_hand'][1] < 795:
        return True
    
    # if hand_bbox_list[0]['right_hand'][0] > 1610:
    #     return True
    
    # if hand_bbox_list[0]['right_hand'][1] > 1336:
    #     return True

    return False

def tracker(prev_box, prev_frame, curr_frame):
    prev_box = np.array(prev_box, dtype=np.float32)
    prev_pts = prev_box.reshape(1, -1, 2)

    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    curr_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_frame_gray, curr_frame_gray, prev_pts, None, maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    x, y, w, h = prev_box.reshape(4)
    curr_box = np.array([x + w * (curr_pts[0][0][0] - prev_pts[0][0][0]),
                        y + h * (curr_pts[0][0][1] - prev_pts[0][0][1]),
                        w, h], dtype=np.float32)

    return curr_box
