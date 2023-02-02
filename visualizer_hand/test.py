import torch
import mano
from mano.utils import Mesh
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type = str, default = 'mocap_output/mocap', help = 'pkl file directory')
parser.add_argument('--output_dir', type = str, default = 'visualized_seq', help = 'output location directory')

if __name__ == "__main__":
    args = parser.parse_args()
    model_path = 'visualizer_hand/models/mano'

    # n_comps = 45
    # batch_size = 10
    directory = args.input_dir 
    for pkl in sorted(os.listdir(directory)):
        with open(os.path.join(directory, pkl), 'rb') as f:
            data = pickle.load(f)
            left_betas = torch.from_numpy(data['pred_output_list'][0]['left_hand']['pred_hand_betas'])
            left_pose = torch.from_numpy(data['pred_output_list'][0]['left_hand']['pred_hand_pose'])
            camera = torch.from_numpy(data['pred_output_list'][0]['left_hand']['pred_camera'])
        # rh_model = mano.load(model_path=model_path,
        #                     is_right= True,
        #                     num_pca_comps=npcomps,
        #                     batch_size =batch_size,
        #                     flat_hand_mean=False)

        lh_model = mano.load(model_path=model_path,
                            is_right= False,
                            num_pca_comps=48,
                            batch_size =1,
                            flat_hand_mean=False)

        # betas = torch.rand(batch_size, 10)*.1
        # pose = torch.rand(batch_size, n_comps)*.1
        global_orient = torch.rand(1, 3)
        transl        = torch.rand(1, 3)

        # output_r = rh_model(betas=betas,
        #                 global_orient=global_orient,
        #                 hand_pose=pose,
        #                 transl=transl,
        #                 return_verts=True,
        #                 return_tips = True)
        
        output_l = lh_model(betas=left_betas,
                        global_orient=global_orient,
                        hand_pose=left_pose,
                        transl=transl,
                        return_verts=True,
                        return_tips = True)


        # rh_meshes = rh_model.hand_meshes(output_r)
        # rj_meshes = rh_model.joint_meshes(output_r)

        lh_meshes = lh_model.hand_meshes(output_l)
        lj_meshes = lh_model.joint_meshes(output_l)

        #visualize hand mesh only
        lh_meshes[0].show()

        # #visualize joints mesh only
        # j_meshes[0].show()

        #visualize hand and joint meshes
        # hj_meshes = Mesh.concatenate_meshes([lh_meshes[0], lh_meshes[0]])
        # hj_meshes.show() 