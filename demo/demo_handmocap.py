# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
sys.path.append("/home/yc4ny/frankmocap")
import numpy as np
import cv2
import json
import torch
from torchvision.transforms import Normalize

from demo.demo_options import DemoOptions
import mocap_utils.general_utils as gnu
import mocap_utils.demo_utils as demo_utils

from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector

import renderer.image_utils as imu
from renderer.viewer2D import ImShow
import time
from tracking.track import tracker, check_negative_numbers, check_bbox_consistency
# from optimize.main import optimize

def run_hand_mocap(args, bbox_detector, hand_mocap, visualizer):
    #Set up input data (images or webcam)
    input_type, input_data = demo_utils.setup_input(args)
 
    assert args.out_dir is not None, "Please specify output dir to store the results"
    cur_frame = args.start_frame
    video_frame = 0
    hand_bbox_track = []
    while True:
        # load data
        load_bbox = False

        if input_type =='image_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr  = cv2.imread(image_path)
            else:
                img_original_bgr = None

        elif input_type == 'bbox_dir':
            if cur_frame < len(input_data):
                print("Use pre-computed bounding boxes")
                image_path = input_data[cur_frame]['image_path']
                hand_bbox_list = input_data[cur_frame]['hand_bbox_list']
                body_bbox_list = input_data[cur_frame]['body_bbox_list']
                img_original_bgr  = cv2.imread(image_path)
                load_bbox = True
            else:
                img_original_bgr = None

        elif input_type == 'video':      
            _, img_original_bgr = input_data.read()
            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        
        elif input_type == 'webcam':
            _, img_original_bgr = input_data.read()

            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        else:
            assert False, "Unknown input_type"

        cur_frame +=1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break   
        print("--------------------------------------")

        # bbox detection
        if load_bbox:
            body_pose_list = None
            raw_hand_bboxes = None
        elif args.crop_type == 'hand_crop':
            # hand already cropped, thererore, no need for detection
            img_h, img_w = img_original_bgr.shape[:2]
            body_pose_list = None
            raw_hand_bboxes = None
            hand_bbox_list = [ dict(right_hand = np.array([0, 0, img_w, img_h])) ]
        else:            
            # Input images has other body part or hand not cropped.
            # Use hand detection model & body detector for hand detection
            assert args.crop_type == 'no_crop'
            detect_output = bbox_detector.detect_hand_bbox(img_original_bgr.copy())
            body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = detect_output
        
        # save the obtained body & hand bbox to json file
        if args.save_bbox_output:
            demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)

        # If it is the first frame and no hand detected, use manually fixed bbox
        left_fix = np.array([1483, 932, 1540, 1228], dtype = np.float32)
        right_fix = np.array([3100, 1207, 538, 772], dtype = np.float32)
        if cur_frame == 1: 
            if hand_bbox_list[0]['left_hand'] is None:
                hand_bbox_list[0]['left_hand'] = left_fix
            if hand_bbox_list[0]['right_hand'] is None:
                hand_bbox_list[0]['right_hand'] = right_fix
        
        if cur_frame != 1  and hand_bbox_list[0]['left_hand'] is None:
            print(f"Left hand not deteced, using tracker from previous frame: {image_path}")
            previous_frame = cv2.imread("mocap_output/frames/" + str(cur_frame -2).zfill(5) + ".jpg")
            current_frame = cv2.imread("mocap_output/frames/" + str(cur_frame-1).zfill(5) + ".jpg")
            hand_bbox_list[0]['left_hand'] = tracker(hand_bbox_track[cur_frame-2][0]['left_hand'],previous_frame, current_frame)

        if cur_frame != 1  and hand_bbox_list[0]['right_hand'] is None:
            print(f"Right hand not deteced, using tracker from previous frame: {image_path}")
            previous_frame = cv2.imread("mocap_output/frames/" + str(cur_frame -2).zfill(5) + ".jpg")
            current_frame = cv2.imread("mocap_output/frames/" + str(cur_frame-1).zfill(5) + ".jpg")
            hand_bbox_list[0]['right_hand'] = tracker(hand_bbox_track[cur_frame-2][0]['right_hand'],previous_frame, current_frame)
       
        if cur_frame != 1 and check_negative_numbers(hand_bbox_list) is True:
            hand_bbox_list[0]['left_hand'] = hand_bbox_track[cur_frame-2][0]['left_hand']
            hand_bbox_list[0]['right_hand'] = hand_bbox_track[cur_frame-2][0]['right_hand']
        
        if cur_frame != 1 and check_bbox_consistency(hand_bbox_list[0]['left_hand'], hand_bbox_track[cur_frame-2][0]['left_hand']) is True:
            hand_bbox_list[0]['left_hand'] = hand_bbox_track[cur_frame-2][0]['left_hand']

        if cur_frame != 1 and check_bbox_consistency(hand_bbox_list[0]['right_hand'], hand_bbox_track[cur_frame-2][0]['right_hand']) is True:
            hand_bbox_list[0]['right_hand'] = hand_bbox_track[cur_frame-2][0]['right_hand']
        
        hand_bbox_track.append(hand_bbox_list)
    
        # Hand Pose Regression
        pred_output_list = hand_mocap.regress(
                img_original_bgr, hand_bbox_list, add_margin=True)
        assert len(hand_bbox_list) == len(body_bbox_list)
        assert len(body_bbox_list) == len(pred_output_list)

        # pose, beta = optimize()
        # pred_temp = pred_output_list
        # pred_output_list[0]['left_hand']['pred_hand_pose'][:,3:] = pose 
        # pred_output_list[0]['left_hand']['pred_hand_betas'] = beta 
        # extract mesh for rendering (vertices in image space and faces) from pred_output_list
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        # visualize
        res_img = visualizer.visualize(
            img_original_bgr, 
            pred_mesh_list = pred_mesh_list, 
            hand_bbox_list = hand_bbox_list)

        # show result in the screen
        if not args.no_display:
            res_img = res_img.astype(np.uint8)
            ImShow(res_img)

        # save the image (we can make an option here)
        if args.out_dir is not None:
            demo_utils.save_res_img(args.out_dir, image_path, res_img)

        # save predictions to pkl
        if args.save_pred_pkl:
            demo_type = 'hand'
            demo_utils.save_pred_to_pkl(
                args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)

        print(f"Processed : {image_path}")
        
    #save images as a video
    if not args.no_video_out and input_type in ['video', 'webcam']:
        demo_utils.gen_video_out(args.out_dir, args.seq_name)

    # When everything done, release the capture
    if input_type =='webcam' and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()

  
def main():
    args = DemoOptions().parse()
    # For Debugging 
    args.input_path = 'sample_data/lefthand.mp4'
    args.out_dir = 'mocap_output'
    args.view_type = 'ego_centric'
    args.use_smplx = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    #Set Bbox detector
    bbox_detector =  HandBboxDetector(args.view_type, device)

    # Set Mocap regressor
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device = device)

    # Set Visualizer
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)
    # visualizer = None

    # run
    run_hand_mocap(args, bbox_detector, hand_mocap, visualizer)
   

if __name__ == '__main__':
    main()