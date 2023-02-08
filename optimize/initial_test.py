import torch
import mano
from mano.utils import Mesh
import pickle
import sys
import cv2
sys.path.append('/home/yc4ny/frankmocap')
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
import numpy as np 

def translate(coord, array):
    # calculate the difference between the input coordinate and the 5th coordinate in the array
    x_diff = coord[0] - array[4][0]
    y_diff = coord[1] - array[4][1]
    print("Translation: x = " + str(x_diff) + "\ny = "+ str(y_diff))
    # move all other coordinates in the array by the x and y differences
    for i in range(len(array)):
        array[i][0] += x_diff
        array[i][1] += y_diff
    
    return array

if __name__ == "__main__":
    gt_2d = np.array([
        [2346, 2154], # 0
        [2550, 1580], # 1
        [2501, 1311], # 2
        [2409, 1224], # 3
        [2287, 1501], # 4
        [2205, 1199], # 5
        [2156, 1063], # 6 
        [1838, 1751], # 7
        [1748, 1452], # 8
        [1688, 1289], # 9
        [2058, 1613], # 10
        [1985, 1270], # 11
        [1960, 1101], # 12
        [2926, 2034], # 13
        [2923, 1817], # 14
        [2831, 1727], # 15
        [2749, 1675], # 16
        [2349, 1164], # 17
        [2110, 960], # 18
        [1920, 984], # 19
        [1672, 1128], # 20
    ])
    with open('mocap_output/mocap/00001_prediction_result.pkl', 'rb') as f:
        a = pickle.load(f)

    # Pose (1,48) -> global (1,3), pose (1,45) 
    # Beta (1,10)
    pose = a['pred_output_list'][0]['left_hand']['pred_hand_pose']
    global_orient = torch.from_numpy(pose[0][:3].reshape(1,3))
    pose = torch.from_numpy(pose[0][3:].reshape(1,45))
    beta = torch.from_numpy(a['pred_output_list'][0]['left_hand']['pred_hand_betas'])
    #scale, tranX, tranY
    cam = torch.from_numpy(a['pred_output_list'][0]['left_hand']['pred_camera'].reshape(1,3))
    # translation = torch.from_numpy(np.array([[-0.1,-0.03,0]]))

    model_path = 'optimize/models/mano'
    model = mano.load(model_path=model_path,
                        is_rhand= False,
                        num_pca_comps=45,
                        batch_size=1,
                        flat_hand_mean=False)

    output = model(betas=beta,
                    global_orient=global_orient,
                    hand_pose=pose,
                    transl=None,
                    return_verts=True,
                    return_tips = True)
    
    joints_smplcoord = output[1].cpu().detach().numpy()[0]
    cam = a['pred_output_list'][0]['left_hand']['pred_camera']
    cam_scale = cam[0]
    cam_trans = cam[1:]

    # SMPL space -> bbox space
    joints_bboxcoord = convert_smpl_to_bbox(joints_smplcoord, cam_scale, cam_trans, bAppTransFirst=True)
    hand_boxScale_o2n = a['pred_output_list'][0]['left_hand']['bbox_scale_ratio']
    hand_bboxTopLeft = a['pred_output_list'][0]['left_hand']['bbox_top_left']
    # Bbox space -> original image space
    joints_imgcoord = convert_bbox_to_oriIm(joints_bboxcoord, hand_boxScale_o2n, hand_bboxTopLeft, 3840, 2160) 

    # Translate according to index 4 finger
    joints_imgcoord = translate(gt_2d[4], joints_imgcoord)
    img = cv2.imread("mocap_output/frames/00001.jpg")

    for i in range(joints_imgcoord.shape[0]):
        cv2.circle(img, (int(gt_2d[i][0]), int(gt_2d[i][1])), 10, (0,0,255), -1)
        cv2.putText(img, str(i), (int(gt_2d[i][0]), int(gt_2d[i][1])), fontScale=2, fontFace = cv2.FONT_HERSHEY_COMPLEX, color = (0,0,255), thickness = 3)
        cv2.circle(img, (int(joints_imgcoord[i][0]), int(joints_imgcoord[i][1])), 10, (0,255,0), -1)
        cv2.putText(img, str(i), (int(joints_imgcoord[i][0]), int(joints_imgcoord[i][1])), fontScale=2, fontFace = cv2.FONT_HERSHEY_COMPLEX, color = (0,255,0), thickness = 3)
    
    img = cv2.imwrite("MANO/temp/test.jpg", img)
    

