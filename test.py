import os
import sys
import numpy as np
import argparse
import time
import torch
from torch.utils.data import DataLoader
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.graspnet import GraspNet, pred_decode
# 导入GraspNet官方API，用于评估
from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn
from utils.collision_detector import ModelFreeCollisionDetector



# --- 1. 配置加载：解析命令行参数 ---
# 创建一个参数解析器对象
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default=None, required=True)
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default=None, required=True)
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
# 添加 --collision_thresh 参数，指定碰撞检测中的距离阈值
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
# 添加 --infer 参数，一个标志，如果使用此参数，则执行推
parser.add_argument('--infer', action='store_true', default=False)
# 添加 --eval 参数，一个标志，如果使用此参数，则执行评估
parser.add_argument('--eval', action='store_true', default=False)
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
# --- 全局配置 ---
# 检查用于保存结果的目录是否存在，如果不存在则创建
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)



# Init datasets and dataloaders 
# 为数据加载器中的每个工作进程设置不同的随机种子
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass



# 定义推理函数
def inference():
    # 实例化测试数据集，split='test_seen'表示使用见过的场景进行测试
    test_dataset = GraspNetDataset(cfgs.dataset_root, split='test_seen', camera=cfgs.camera, num_points=cfgs.num_point,
                                   voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False, load_label=False)
    # 打印测试数据集的样本总数
    print('Test dataset length<测试数据集的样本总数>: ', len(test_dataset))
    # 获取场景列表，用于后续保存结果到对应的场景文件夹
    scene_list = test_dataset.scene_list()
    # 创建测试数据加载器，shuffle=False确保数据按顺序加载
    test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False,
                                 num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
    # 打印测试数据加载器的批次总数
    print('Test dataloader length<数据加载器的批次总数>: ', len(test_dataloader))
    
    
    # Init the model
    # --- 模型初始化与加载 ---
    # 实例化GraspNet模型，并设置为非训练模式
    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    # 设置计算设备，如果CUDA可用则使用GPU，否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 将模型的所有参数和缓冲区移动到指定的计算设备
    net.to(device)
    # Load checkpoint
    # 加载预训练的模型检查点文件
    checkpoint = torch.load(cfgs.checkpoint_path)
    # 将检查点中的模型状态字典加载到当前模型中
    net.load_state_dict(checkpoint['model_state_dict'])
    # 获取检查点保存时的周期数
    start_epoch = checkpoint['epoch']
    # 打印加载信息
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))


    # --- 推理循环 ---
    batch_interval = 100  # 定义日志记录间隔，每100个批次记录一次
    # 将模型设置为评估模式（关闭Dropout等）
    net.eval()  # 记录开始时间
     
    # 遍历测试数据加载器的每个批次
    tic = time.time()
    for batch_idx, batch_data in enumerate(test_dataloader):
        # 将批次数据中的所有张量移动到计算设备（GPU/CPU）
        for key in batch_data:
            if 'list' in key:# 特殊处理列表嵌套的张量
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:  # 处理普通张量
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        # 使用torch.no_grad()上下文管理器，在此区域内不计算梯度，以节省内存和加速
        with torch.no_grad():
            # 前向传播：将数据送入网络，得到包含预测结果的end_points字典
            end_points = net(batch_data)
            # 解码：将网络的原始输出转换为结构化的抓取姿态表示
            grasp_preds = pred_decode(end_points)


        # --- 保存结果用于评估 ---
        # 遍历批次内的每个样本（通常batch_size为1）
        # Dump results for evaluation
        for i in range(cfgs.batch_size):
            # 计算当前数据在整个数据集中的索引
            data_idx = batch_idx * cfgs.batch_size + i
            # 获取当前样本的预测结果，并将其转换为numpy数组
            preds = grasp_preds[i].detach().cpu().numpy()


            # 使用预测结果创建一个GraspGroup对象
            gg = GraspGroup(preds)
            # collision detection
            # 如果设置了碰撞阈值，则执行碰撞检测
            if cfgs.collision_thresh > 0:
                # 获取当前样本的原始点云数据
                cloud = test_dataset.get_data(data_idx, return_raw_cloud=True)
                # 初始化无模型碰撞检测器
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
                # 检测哪些抓取会发生碰撞，返回一个布尔掩码
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                # 使用掩码的反转（~）来保留所有未发生碰撞的抓取
                gg = gg[~collision_mask]


            # save grasps
            # 构建保存抓取结果的目录和文件路径
            save_dir = os.path.join(cfgs.dump_dir, scene_list[data_idx], cfgs.camera)
            save_path = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.npy')
            # 如果保存目录不存在，则创建它
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 将过滤后的GraspGroup对象保存为.npy文件
            gg.save_npy(save_path)

        # 每处理一定数量的批次后，打印进度信息
        if (batch_idx + 1) % batch_interval == 0:
            toc = time.time()   # 记录当前时间
            print('Eval batch: %d, time: %fs' % (batch_idx + 1, (toc - tic) / batch_interval))
            tic = time.time()     # 重置开始时间



# 定义评估函数
def evaluate(dump_dir):
    # 初始化GraspNet评估器
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test_seen')
    # 调用评估函数，对保存在dump_dir中的所有预测结果进行评估，返回AP结果
    res, ap = ge.eval_seen(dump_folder=dump_dir, proc=6)
    # 构建保存AP结果的文件路径
    save_dir = os.path.join(cfgs.dump_dir, 'ap_{}.npy'.format(cfgs.camera))
    # 将AP结果保存为.npy文件
    np.save(save_dir, res)


if __name__ == '__main__':
    # 如果命令行中使用了 --infer 参数，则执行推理函数
    if cfgs.infer:
        inference()
    # 如果命令行中使用了 --eval 参数，则执行评估函数
    if cfgs.eval:
        evaluate(cfgs.dump_dir)
