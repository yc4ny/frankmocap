import torch
import mano
from mano.utils import Mesh

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import pickle 

def plot_3d_points(input_points):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Convert the input tensor to a numpy array
    points = np.array(input_points[0])

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ax.scatter3D(x, y, z, c=z, cmap='Reds')
    
    for i, point in enumerate(points):
        ax.text(point[0], point[1], point[2], str(i))

    plt.show()

if __name__ == "__main__":
    with open("optimize/output_parameters/014999.pkl",'rb') as file:
        parameters = pickle.load(file)
    beta = torch.Tensor(parameters['beta'])
    global_orient = torch.Tensor(parameters['global_orient'])
    pose = torch.Tensor(parameters['pose'])
    cam = torch.Tensor(parameters['cam'])

    # with open('mocap_output/mocap/01422_prediction_result.pkl', 'rb') as f:
    #     a = pickle.load(f)

    # pose = a['pred_output_list'][0]['left_hand']['pred_hand_pose']
    # global_orient = torch.from_numpy(pose[0][:3].reshape(1,3))
    # # global_orient = torch.Tensor([[0, 0, 0 ]])
    # pose = torch.from_numpy(pose[0][3:].reshape(1,45))
    # beta = torch.from_numpy(a['pred_output_list'][0]['left_hand']['pred_hand_betas'])


    model_path = 'optimize/models/mano'
    n_comps = 45
    batch_size = 1
    rh_model = mano.load(model_path=model_path,
                        is_rhand= False,
                        num_pca_comps=n_comps,
                        batch_size=batch_size,
                        flat_hand_mean=False)

    output = rh_model(betas=beta,
                    global_orient=global_orient,
                    hand_pose=pose,
                    transl=None,
                    return_verts=True,
                    return_tips = True)
    
    # plot_3d_points(output[1])

    
    h_meshes = rh_model.hand_meshes(output)
    # j_meshes = rh_model.joint_meshes(output)
    #visualize hand mesh only
    h_meshes[0].show()

    # #visualize joints mesh only
    # j_meshes[0].show()

    # #visualize hand and joint meshes
    # hj_meshes = Mesh.concatenate_meshes([h_meshes[0], j_meshes[0]])
    # hj_meshes.show() 
