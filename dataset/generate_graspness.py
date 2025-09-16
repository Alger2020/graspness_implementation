import numpy as np
import os
from PIL import Image
import scipy.io as scio
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image
from knn.knn_modules import knn
import torch
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import get_obj_pose_list, transform_points
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--camera_type', default='kinect', help='Camera split [realsense/kinect]')

'''
这个脚本的主要功能是为 GraspNet 数据集中的每个场景点云生成“抓取度”（graspness）标签。
抓取度表示一个点云中的点有多适合作为抓取点。它通过将预先计算好的物体模型上的抓取标签，投影到实际的场景点云中来实现。
'''

if __name__ == '__main__':
    cfgs = parser.parse_args()
    dataset_root = cfgs.dataset_root   # set dataset root
    camera_type = cfgs.camera_type   # kinect / realsense
    save_path_root = os.path.join(dataset_root, 'graspness') #用于指定生成的抓取度标签的保存根目录

    num_views, num_angles, num_depths = 300, 12, 4 #定义了每个抓取点的抓取姿态数量（300个视角，每个视角12个旋转角度，4个深度）。
    fric_coef_thresh = 0.8  #摩擦系数的阈值，用于过滤掉不稳定的抓取。
    point_grasp_num = num_views * num_angles * num_depths  #计算每个点上总的可能抓取姿态数量。
    
    
    '''
    开始遍历场景（从场景0到99）。
    为每个场景构建保存路径，如果路径不存在则创建它。
    加载该场景预先计算好的碰撞标签。这些标签指明了哪些抓取姿态会与场景中的其他物体发生碰撞。
    将碰撞标签存入 collision_dump 列表中。
    '''
    # for scene_id in range(100):
    for scene_id in tqdm(range(100), desc="Processing Scenes"):
        save_path = os.path.join(save_path_root, 'scene_' + str(scene_id).zfill(4), camera_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        labels = np.load(
            os.path.join(dataset_root, 'collision_label', 'scene_' + str(scene_id).zfill(4), 'collision_labels.npz'))
        collision_dump = []
        for j in range(len(labels)):
            collision_dump.append(labels['arr_{}'.format(j)])



        
        #遍历每个场景下的256个不同视角（标注）。
        # for ann_id in range(256):
        for ann_id in tqdm(range(256), desc=f"Scene {scene_id:04d}", leave=False):       
            
            # get scene point cloud
            # 打印当前处理的场景ID和标注ID。
            # 加载深度图、语义分割图和元数据（.mat文件）。
            # 从元数据中提取相机内参和深度因子。
            # 创建一个 CameraInfo 对象来存储相机参数。
            # 使用深度图和相机信息生成场景的原始点云。
            # print('generating scene: {} ann: {}'.format(scene_id, ann_id))
            depth = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                     camera_type, 'depth', str(ann_id).zfill(4) + '.png')))
            seg = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                   camera_type, 'label', str(ann_id).zfill(4) + '.png')))
            meta = scio.loadmat(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                             camera_type, 'meta', str(ann_id).zfill(4) + '.mat'))
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
            camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                                factor_depth)
            cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
            
            
            

            # remove outlier and get objectness label
            #对点云进行预处理。
                #创建深度掩码，过滤掉深度值为0的点。
                #加载相机位姿，并获取当前标注ID对应的位姿。
                #加载相机相对于桌面的对齐矩阵。
                #计算最终的变换矩阵 trans。
                #使用 get_workspace_mask 函数移除工作区外的点和离群点。
                #合并深度掩码和工作区掩码，得到最终的有效点掩码 mask。
                #使用掩码过滤点云，得到 cloud_masked。
                #同时，使用该掩码从分割图中提取对应点的物体标签 objectness_label。
            depth_mask = (depth > 0)
            camera_poses = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                camera_type, 'camera_poses.npy'))
            camera_pose = camera_poses[ann_id]
            align_mat = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                             camera_type, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_pose)
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
            cloud_masked = cloud[mask]
            objectness_label = seg[mask]





            # get scene object and grasp info
            #获取场景中每个物体的抓取信息。
               #读取XML标注文件，获取场景中所有物体的位姿信息。
               #获取物体ID列表 obj_list 和它们对应的位姿列表 pose_list。
               #遍历场景中的每个物体，加载其预先计算好的抓取标签（包括抓取点、偏移量和摩擦系数）。
               #将这些标签存储在 grasp_labels 字典中，以物体ID为键。
            scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                  camera_type, 'annotations', '%04d.xml' % ann_id))
            pose_vectors = scene_reader.getposevectorlist()
            obj_list, pose_list = get_obj_pose_list(camera_pose, pose_vectors)
            grasp_labels = {}
            for i in obj_list:
                file = np.load(os.path.join(dataset_root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
                grasp_labels[i] = (file['points'].astype(np.float32), file['offsets'].astype(np.float32),
                                   file['scores'].astype(np.float32))





            #计算物体模型上每个采样点的抓取度。
                #遍历场景中的每个物体及其位姿。
                #获取该物体的抓取标签和碰撞信息。
                #创建一个 valid_grasp_mask，它标记了所有有效的抓取（摩擦系数在阈值内且不发生碰撞）。
                #通过计算有效抓取的比例，得到每个采样点的 graspness（抓取度）分数。
                #将物体模型上的采样点 sampled_points 变换到相机坐标系下。
                #将变换后的点和它们对应的抓取度分数分别存入列表。
                #最后，将列表堆叠成Numpy数组
            grasp_points = []
            grasp_points_graspness = []
            for i, (obj_idx, trans_) in enumerate(zip(obj_list, pose_list)):
                sampled_points, offsets, fric_coefs = grasp_labels[obj_idx]
                collision = collision_dump[i]  # Npoints * num_views * num_angles * num_depths
                num_points = sampled_points.shape[0]

                valid_grasp_mask = ((fric_coefs <= fric_coef_thresh) & (fric_coefs > 0) & ~collision)
                valid_grasp_mask = valid_grasp_mask.reshape(num_points, -1)
                graspness = np.sum(valid_grasp_mask, axis=1) / point_grasp_num
                target_points = transform_points(sampled_points, trans_)
                target_points = transform_points(target_points, np.linalg.inv(camera_pose))  # fix bug
                grasp_points.append(target_points)
                grasp_points_graspness.append(graspness.reshape(num_points, 1))
            grasp_points = np.vstack(grasp_points)
            grasp_points_graspness = np.vstack(grasp_points_graspness)


            #将物体模型上的抓取度传播到场景点云上。
                #将带有抓取度分数的点 grasp_points 和分数 grasp_points_graspness 转换为PyTorch张量并移至GPU。
                #初始化一个零数组 cloud_masked_graspness 用于存储场景点云的抓取度。
                #由于显存限制，将场景点云 cloud_masked 分成10000个点的小块进行处理。
                #在循环中，对每个点云块：
                #使用 knn 算法，为块中的每个点在 grasp_points（带有抓取度分数的点集）中找到最近的一个点。
                #使用找到的最近点的索引 nn_inds，从 grasp_points_graspness 中取出对应的抓取度分数。
                #将这个分数赋给当前处理的点云块
            grasp_points = torch.from_numpy(grasp_points).cuda()
            grasp_points_graspness = torch.from_numpy(grasp_points_graspness).cuda()
            grasp_points = grasp_points.transpose(0, 1).contiguous().unsqueeze(0)

           
            masked_points_num = cloud_masked.shape[0]
            cloud_masked_graspness = np.zeros((masked_points_num, 1))
            part_num = int(masked_points_num / 10000)
            for i in range(1, part_num + 2):   # lack of cuda memory
                if i == part_num + 1:
                    cloud_masked_partial = cloud_masked[10000 * part_num:]
                    if len(cloud_masked_partial) == 0:
                        break
                else:
                    cloud_masked_partial = cloud_masked[10000 * (i - 1):(i * 10000)]
                cloud_masked_partial = torch.from_numpy(cloud_masked_partial).cuda()
                
                
                
               
                cloud_masked_partial = cloud_masked_partial.transpose(0, 1).contiguous().unsqueeze(0)
                nn_inds = knn(grasp_points, cloud_masked_partial, k=1).squeeze() - 1
                cloud_masked_graspness[10000 * (i - 1):(i * 10000)] = torch.index_select(
                    grasp_points_graspness, 0, nn_inds).cpu().numpy()
               
                

            #后处理并保存结果。
                #对整个场景点云的抓取度分数进行归一化，使其范围在0到1之间。
                #将最终生成的抓取度标签数组保存为 .npy 文件，文件名与标注ID对应。
            max_graspness = np.max(cloud_masked_graspness)
            min_graspness = np.min(cloud_masked_graspness)
            cloud_masked_graspness = (cloud_masked_graspness - min_graspness) / (max_graspness - min_graspness)

            np.save(os.path.join(save_path, str(ann_id).zfill(4) + '.npy'), cloud_masked_graspness)
