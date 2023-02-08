import torch
import sys
sys.path.append('/home/yc4ny/frankmocap/optimize')
import mano
from mano.utils import Mesh
import pickle
import cv2
import numpy as np 
import torch.optim as optim
sys.path.append('/home/yc4ny/frankmocap')
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
import numpy as np 
import pickle 
import os
from tqdm import tqdm
import time

def translate(coord, array):
    # calculate the difference between the input coordinate and the 5th coordinate in the array
    x_diff = coord[0] - array[4][0]
    y_diff = coord[1] - array[4][1]
    # move all other coordinates in the array by the x and y differences
    for i in range(len(array)):
        array[i][0] += x_diff
        array[i][1] += y_diff
    
    return array

def mse(x_hat, x, a, cam):
    # cam =torch.Tensor(a['pred_output_list'][0]['left_hand']['pred_camera'])
    cam_scale = cam[0][0]
    cam_trans = cam[0][1:]
    hand_boxScale_o2n = torch.Tensor([a['pred_output_list'][0]['left_hand']['bbox_scale_ratio']])
    hand_bboxTopLeft = torch.Tensor(a['pred_output_list'][0]['left_hand']['bbox_top_left'])
    # Convert mano to bbox coordinates
    x_hat[0][:,0:2] = torch.add(x_hat[0][:,0:2], cam_trans)
    x_hat = torch.mul(x_hat, cam_scale)
    x_hat = torch.mul(x_hat, 224*0.5)

    # Convert bbox to original image
    x_hat = torch.div(x_hat, hand_boxScale_o2n)
    x_hat[0][:,:2] = torch.add(x_hat[0][:,0:2], (torch.add(hand_bboxTopLeft, torch.div(224*0.5, hand_boxScale_o2n))))
    x_hat = x_hat[0][:,:2]
    # Translation 
    x_diff = torch.sub(x[4][0], x_hat[4][0])
    y_diff = torch.sub(x[4][1], x_hat[4][1])
    x_hat[:,:1] = torch.sub(x_hat[:,:1], torch.mul(x_diff,-1))
    x_hat[:,1:] = torch.sub(x_hat[:,1:], torch.mul(y_diff,-1))
    y = ((x - x_hat)**2).mean()
    
    return y

if __name__ == "__main__":

    gt_2d = torch.tensor([
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
    ], dtype = torch.float32)
    with open('mocap_output/mocap/00000_prediction_result.pkl', 'rb') as f:
        a = pickle.load(f)

    pose = a['pred_output_list'][0]['left_hand']['pred_hand_pose']
    global_orient = torch.from_numpy(pose[0][:3].reshape(1,3))
    # global_orient = torch.Tensor([[0, 0, 0 ]])
    pose = torch.from_numpy(pose[0][3:].reshape(1,45))
    beta = torch.from_numpy(a['pred_output_list'][0]['left_hand']['pred_hand_betas'])
    cam = torch.from_numpy(a['pred_output_list'][0]['left_hand']['pred_camera'].reshape(1,3))

    model = mano.load(model_path='optimize/models/mano',
                        is_rhand= False,
                        num_pca_comps=45,
                        batch_size=1,
                        flat_hand_mean=False)
    # # Define the loss function
    # criterion = torch.nn.MSELoss()
    # Define the optimizer
    optimizer = optim.Adam([beta, pose, global_orient, cam], lr=0.001)
    if not os.path.exists("optimize/frames"):
        os.makedirs("optimize/frames")

    # Draw GT joints on image to avoid loops
    connections = [[7, 8], [8, 9], [9, 20], [7, 0], [0, 10], [10, 11], [11, 12], [12, 19], [0, 4], [4, 5], [5, 6], [6, 18], [0, 1], [1, 2], [2, 3], [3, 17], [0, 13], [13, 14], [14, 15], [15, 16]]
    
    # cv2.putText(img, str(i), (int(gt_2d[i][0]), int(gt_2d[i][1])), fontScale=2, fontFace = cv2.FONT_HERSHEY_COMPLEX, color = (0,0,255), thickness = 3)
    # For printing loss on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 3
    text_thickness = 2
    text_size, _ = cv2.getTextSize("Loss", font, text_scale, text_thickness)
    
    # Loop over a set of iterations (epochs)
    start_time = time.time()
    for epoch in tqdm(range(15000)):
        beta.requires_grad_()
        pose.requires_grad_()
        global_orient.requires_grad_()
        cam.requires_grad_()
        # Zero the gradients
        optimizer.zero_grad()

        # Get the predicted 3D joints
        output = model(betas=beta,
                        global_orient=global_orient,
                        hand_pose=pose,
                        transl=None,
                        return_verts=True,
                        return_tips = True)

        loss = mse(output.joints, gt_2d, a, cam)
        # Calculate the gradients
        loss.backward()
        # Update the beta and hand_pose parameters
        optimizer.step()

        with open(f'optimize/output_parameters/{str(epoch).zfill(6)}.pkl', 'wb') as f:
            pickle.dump({'beta': beta.detach().numpy(), 
                        'pose': pose.detach().numpy(), 
                        'global_orient': global_orient.detach().numpy(),
                        'cam': cam.detach().numpy(),
                        'loss': loss.item()
                        }, f)
        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')   
    end_time = time.time()
    time_interval = end_time - start_time
    print("Optimization Time: " + str(time_interval) + "s")

    
    # Visualize optimization 
    print("-------------------------------------------------------")
    print("Saving visualization of optimization process")
    path = "optimize/output_parameters"
    parameters_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
    num_files = len(parameters_files)
    count = 0 
    for epoch in tqdm(range(num_files)):
        if epoch % 50 == 0:
            filename = f"{path}/{str(epoch).zfill(6)}.pkl"
            with open(filename,'rb') as file:
                parameters = pickle.load(file)
            beta = torch.Tensor(parameters['beta'])
            global_orient = torch.Tensor(parameters['global_orient'])
            pose = torch.Tensor(parameters['pose'])
            cam = torch.Tensor(parameters['cam'])
            loss = parameters['loss']
            output = model(beta, global_orient, pose, transl = None, return_verts = True, return_tips = True)
            joints_smplcoord = output[1].cpu().detach().numpy()[0]
            cam_scale = cam[0][0].numpy()
            cam_trans = cam[0][1:].numpy()
            # SMPL space -> bbox space
            joints_bboxcoord = convert_smpl_to_bbox(joints_smplcoord, cam_scale, cam_trans, bAppTransFirst=True)
            hand_boxScale_o2n = a['pred_output_list'][0]['left_hand']['bbox_scale_ratio']
            hand_bboxTopLeft = a['pred_output_list'][0]['left_hand']['bbox_top_left']
            # Bbox space -> original image space
            joints_imgcoord = convert_bbox_to_oriIm(joints_bboxcoord, hand_boxScale_o2n, hand_bboxTopLeft, 3840, 2160) 
            # Translate according to index 4 finger
            joints_imgcoord = translate(gt_2d[4], joints_imgcoord)

            #Visualize Inference
            img = cv2.imread("mocap_output/frames/00000.jpg")
            top_right_corner = (img.shape[1] - text_size[0] - 2000, text_size[1] + 10)
            location_time =(img.shape[1] - text_size[0] - 2000, text_size[1] + 100)
            for i, j in connections:
                cv2.line(img, (int(gt_2d[i][0]), int(gt_2d[i][1])), (int(gt_2d[j][0]), int(gt_2d[j][1])), (0, 0,255), 5)
            for i in range(joints_imgcoord.shape[0]):
                cv2.circle(img, (int(gt_2d[i][0]), int(gt_2d[i][1])), 15, (0,0,255), -1)
            # Draw optimized joints on image
            for i in range(joints_imgcoord.shape[0]):
                cv2.circle(img, (int(joints_imgcoord[i][0]), int(joints_imgcoord[i][1])), 15, (0,255,0), -1)
                # cv2.putText(img, str(i), (int(joints_imgcoord[i][0]), int(joints_imgcoord[i][1])), fontScale=2, fontFace = cv2.FONT_HERSHEY_COMPLEX, color = (0,255,0), thickness = 3)
            for i, j in connections:
                cv2.line(img, (int(joints_imgcoord[i][0]), int(joints_imgcoord[i][1])), (int(joints_imgcoord[j][0]), int(joints_imgcoord[j][1])), (0, 255, 0), 5)
            cv2.putText(img,"Iter: " + str(epoch) + " MSE Loss:" + str(round(loss,5)), top_right_corner, font, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
            cv2.putText(img,"Optimization time(15000 iter): " + str(round(time_interval,5)) + "s", location_time, font, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
            cv2.imwrite("optimize/frames/{:04}.jpg".format(count), img)
            count += 1 

    cmd = "ffmpeg -r 30 -i optimize/frames/%04d.jpg -vcodec libx264 -pix_fmt yuv420p -y optimize/vis.mp4"
    os.system(cmd)
    cmd  = "rm -rf optimize/frames/*.jpg"
    os.system(cmd)