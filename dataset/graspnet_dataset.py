""" GraspNet dataset processing.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image

# --- 添加以下代码块 ---
# 将项目根目录添加到Python路径中，以解决模块导入问题
# __file__ 是当前文件路径: .../graspness_implementation/dataset/graspnet_dataset.py
# os.path.dirname(__file__) 是当前文件所在目录: .../graspness_implementation/dataset
# os.path.dirname(os.path.dirname(__file__)) 是上级目录: .../graspness_implementation
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
# --------------------

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
import MinkowskiEngine as ME
from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, get_workspace_mask





'''这个文件的核心作用是创建一个标准的 PyTorch Dataset 类，专门用于加载和预处理 GraspNet 数据集。
它负责从磁盘读取深度图、分割图、位姿信息和预先计算好的抓取标签，
然后将它们转换成神经网络模型（特别是使用了 MinkowskiEngine 的稀疏卷积网络）可以处理的格式，即点云和相关的标签。
'''
class GraspNetDataset(Dataset):
    def __init__(self, root, grasp_labels=None, camera='kinect', split='train', num_points=20000,
                 voxel_size=0.005, remove_outlier=True, augment=False, load_label=True):
        assert (num_points <= 50000)
        self.root = root                          # root: 数据集根目录
        self.split = split                        # 数据集分割：'train', 'test', 'test_seen'等
        self.voxel_size = voxel_size              # 体素大小，用于MinkowskiEngine的量化
        self.num_points = num_points              # 采样的点云数量 (默认 20000)
        self.remove_outlier = remove_outlier      # 是否移除工作区之外的离群点
        self.grasp_labels = grasp_labels          # 抓取标签 (从外部传入)
        self.camera = camera                      # 相机类型('kinect' 或 'realsense') 
        self.augment = augment                    # 是否进行数据增强
        self.load_label = load_label              # 是否加载标签（训练时为True，推理时为False
        self.collision_labels = {}                # 用于存储碰撞标签的字典

        # 根据 'split' 参数确定要加载的场景ID
        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        # 初始化用于存储各种文件路径的列表
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        
        # 遍历所有场景ID，构建每个数据样本的文件路径       
        for x in tqdm(self.sceneIds, desc='<Graspnetdataset init>Loading data path and collision labels...'):
            for img_num in range(256):
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, str(img_num).zfill(4) + '.npy'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            # 如果是训练模式，调用get_data_label获取数据和标签
            return self.get_data_label(index)
        else:
            # 如果是推理模式，只获取数据
            return self.get_data(index)


    # 只获取数据（用于推理）
    def get_data(self, index, return_raw_cloud=False):
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]

        if return_raw_cloud:
            return cloud_masked
        # sample points random
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    }
        return ret_dict


# 主要的方法  get_data_label(index) -获取带标签的数据

#   1.图像加载: 读取RGB、深度、分割图像和元数据
#   2.点云生成: 从深度图像生成3D点云
#   3.点云采样: 随机采样到指定点数
#   4.标签处理: 加载抓取点、偏移、分数和容差标签
#   5.碰撞检测: 处理碰撞标签，将有碰撞的抓取点分数设为0
#   6.可见性过滤: 移除不可见的抓取点

    def get_data_label(self, index):
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        graspness = np.load(self.graspnesspath[index])  # for each point in workspace masked point cloud
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        seg_masked = seg[mask]

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        seg_sampled = seg_masked[idxs]
        graspness_sampled = graspness[idxs]
        objectness_label = seg_sampled.copy()

        objectness_label[objectness_label > 1] = 1

        object_poses_list = []
        grasp_points_list = []
        grasp_widths_list = []
        grasp_scores_list = []
        for i, obj_idx in enumerate(obj_idxs):
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, widths, scores = self.grasp_labels[obj_idx]
            collision = self.collision_labels[scene][i]  # (Np, V, A, D)

            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_widths_list.append(widths[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)

        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),               #(20000>,3),3D点云,N是点云数量，3为xyz
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,    #(20000,3)用于 MinkowskiEngine 的体素坐标，通过将点云坐标除以体素大小得到。
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),        #(20000,3)点云的初始特征，这里简单地设为全1。
                    'graspness_label': graspness_sampled.astype(np.float32),        #(20000,1)点级别的抓取度标签。
                    'objectness_label': objectness_label.astype(np.int64),          #(20000)点级别的物体性标签。区分物体是物体(1)还是背景(0)
                    'object_poses_list': object_poses_list,                         #(9,3,4)物体姿态列表，列表长度为该场景物体数<9>，姿态形状为(3,4)
                    'grasp_points_list': grasp_points_list,                         #(9,300,3)抓取点列表长度为<9>,形状为(300,3) 300是单物体抓取点数量，3是xyz坐标
                    'grasp_widths_list': grasp_widths_list,                         #(9,300,300,12,4)抓取点宽度列表
                    'grasp_scores_list': grasp_scores_list}                         #(9,300,300,12,4)抓取分数列表长度: 9  单个分数形状: (300, 300, 12, 4)   300是预设的视角点，12是预设的旋转角度，4是预设的深度，妈的
        return ret_dict


def load_grasp_labels(root):
     # 定义要加载的物体ID范围
    obj_names = list(range(1, 89))   
     #初始化一个空字典用于存储标签
    grasp_labels = {}
    for obj_name in tqdm(obj_names, desc='<load_grasp_labels>Loading grasping labels...'):
        #构建每个物体标签文件的路径,加载.npz文件
        label = np.load(os.path.join(root, 'grasp_label_simplified', '{}_labels.npz'.format(str(obj_name - 1).zfill(3))))
        #提取需要的数据，并存入字典
        grasp_labels[obj_name] = (label['points'].astype(np.float32), label['width'].astype(np.float32),
                                  label['scores'].astype(np.float32))

    return grasp_labels


def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data])
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch, features_batch, return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original
    }

    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats':
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
    res = collate_fn_(list_data)

    return res





if __name__ == "__main__":
    # --- 1. 初始化和加载 ---
    # 设置数据集根目录
    root = '/T13/jing/graspnet-baseline/dataset/graspnet' # 请确保路径正确

    print("正在加载抓取标签...")
    # GraspNetDataset需要预先加载的抓取标签
    grasp_labels = load_grasp_labels(root) 
    print("抓取标签加载完毕。")

    print("\n正在初始化GraspNetDataset...")
    # 初始化数据集
    train_dataset = GraspNetDataset(root, grasp_labels, split='train')
    print("数据集初始化完毕。")

    # --- 2. 获取单个样本数据 ---
    sample_index = 1 # 选择第一个样本进行检查
    print(f"\n正在从数据集中获取索引为 {sample_index} 的样本...")
    ret_dict = train_dataset[sample_index] # 等同于 train_dataset.get_data_label(sample_index)
    print("样本获取成功！")

    # --- 3. 详细打印 ret_dict 内容 ---
    print("\n----------- 检查 get_data_label 返回的 ret_dict -----------")

    # # 1. 点云坐标 (point_clouds)
    # key = 'point_clouds'
    # data = ret_dict[key]
    # print(f"\n1. {key}:")
    # print(f"   - 形状: {data.shape}")
    # print(f"   - 数据类型: {data.dtype}")
    # print(f"   - 解释: 采样后的场景点云坐标 (x, y, z)。数量为 num_points (默认15000)。")

    # # 2. 体素坐标 (coors)
    # key = 'coors'
    # data = ret_dict[key]
    # print(f"\n2. {key}:")
    # print(f"   - 形状: {data.shape}")
    # print(f"   - 数据类型: {data.dtype}")
    # print(f"   - 解释: 用于MinkowskiEngine的体素化坐标，由 point_clouds / voxel_size 得到。")

    # 3. 点云特征 (feats)
    key = 'feats'
    data = ret_dict[key]
    print(f"\n3. {key}:")
    print(f"   - 形状: {data.shape}")
    print(f"   - 数据类型: {data.dtype}")
    print(f"   - 解释: 点云的初始特征，这里简单地设为全1。")
    print(f"   - 内容预览 (前5个): \n{data[:5]}")

    # 5. 物体性标签 (objectness_label)
    key = 'objectness_label'
    data = ret_dict[key]
    print(f"\n5. {key}:")
    print(f"   - 形状: {data.shape}")
    print(f"   - 数据类型: {data.dtype}")
    print(f"   - 解释: 点级别的监督标签，用于区分点是属于物体(1)还是背景(0)。")
    unique, counts = np.unique(data, return_counts=True)
    print(f"   - 内容统计: {dict(zip(unique, counts))}")
    print(f"   - 内容预览 (前20个): {data[:20]}")
    
    # # 4. 抓取度标签 (graspness_label)
    # key = 'graspness_label'
    # data = ret_dict[key]
    # print(f"\n4. {key}:")
    # print(f"   - 形状: {data.shape}")
    # print(f"   - 数据类型: {data.dtype}")
    # print(f"   - 解释: 点级别的监督标签，表示每个点作为抓取点的“优良程度”。")

    # # 5. 物体性标签 (objectness_label)
    # key = 'objectness_label'
    # data = ret_dict[key]
    # print(f"\n5. {key}:")
    # print(f"   - 形状: {data.shape}")
    # print(f"   - 数据类型: {data.dtype}")
    # print(f"   - 解释: 点级别的监督标签，用于区分点是属于物体(1)还是背景(0)。")

    # # 6. 物体位姿列表 (object_poses_list)
    # key = 'object_poses_list'
    # data_list = ret_dict[key]
    # print(f"\n6. {key}:")
    # print(f"   - 类型: 列表 (list)")
    # if data_list:
    #     print(f"   - 列表长度 (场景中的物体数): {len(data_list)}")
    #     print(f"   - 单个元素(位姿矩阵)的形状: {data_list[0].shape}")
    #     print(f"   - 解释: 列表中每个元素是一个物体的位姿矩阵(3x4)，用于将物体坐标系下的点变换到相机坐标系。")
    # else:
    #     print("   - 场景中没有检测到物体。")

    # # 7. 抓取点列表 (grasp_points_list)
    # key = 'grasp_points_list'
    # data_list = ret_dict[key]
    # print(f"\n7. {key}:")
    # print(f"   - 类型: 列表 (list)")
    # if data_list:
    #     print(f"   - 列表长度: {len(data_list)}")
    #     print(f"   - 单个元素(抓取点集)的形状: {data_list[0].shape}")
    #     print(f"   - 解释: 列表中每个元素对应一个物体的抓取点坐标(x,y,z)，这些点位于物体坐标系下。")
    # else:
    #     print("   - 场景中没有物体，无抓取点。")

    # # 8. 抓取宽度列表 (grasp_widths_list)
    # key = 'grasp_widths_list'
    # data_list = ret_dict[key]
    # print(f"\n8. {key}:")
    # print(f"   - 类型: 列表 (list)")
    # if data_list:
    #     print(f"   - 列表长度: {len(data_list)}")
    #     print(f"   - 单个元素(抓取宽度集)的形状: {data_list[0].shape}")
    #     print(f"   - 解释: 列表中每个元素对应一个物体的抓取宽度，与抓取点一一对应。")
    # else:
    #     print("   - 场景中没有物体，无抓取宽度。")

    # # 9. 抓取分数列表 (grasp_scores_list)
    # key = 'grasp_scores_list'
    # data_list = ret_dict[key]
    # print(f"\n9. {key}:")
    # print(f"   - 类型: 列表 (list)")
    # if data_list:
    #     print(f"   - 列表长度: {len(data_list)}")
    #     print(f"   - 单个元素(抓取分数集)的形状: {data_list[0].shape}")
    #     print(f"   - 解释: 列表中每个元素是对应物体的一组抓取质量分数。形状为(抓取点数, 视角数, 旋转数, 深度数)。其中碰撞的抓取分数已被设为0。")
    # else:
    #     print("   - 场景中没有物体，无抓取分数。")

    print("\n----------- 检查完毕 -----------")