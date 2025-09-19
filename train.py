import os
import sys
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# --- 路径设置 ---
# 获取当前脚本文件(train.py)所在的绝对路径的目录名，即项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))


# --- 从项目其他文件中导入自定义模块 ---
from models.graspnet import GraspNet
from models.loss import get_loss
from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn, load_grasp_labels



'''train.py 脚本核心功能
这个脚本是整个项目的入口，负责执行神经网络的训练流程。它完成了以下几件核心任务：

配置加载：解析命令行参数，设置学习率、批量大小、数据集路径等超参数。
数据准备：初始化 GraspNetDataset，创建 DataLoader 来批量加载和预处理数据。
模型构建：实例化 GraspNet 模型，并设置优化器（Adam）。
训练循环：迭代地将数据送入模型进行前向传播，计算损失，然后反向传播更新模型权重。
日志与保存：使用 TensorBoard 记录训练过程中的损失和精度，并定期保存模型的检查点（checkpoint）。
'''

# --- 1. 配置加载：解析命令行参数 ---
# 创建一个参数解析器对象
parser = argparse.ArgumentParser()
# 添加 --dataset_root 参数，必需，用于指定数据集的根目录
parser.add_argument('--dataset_root', default=None, required=True)
# 添加 --camera 参数，用于选择相机类型（'kinect' 或 'realsense'）
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
# 添加 --checkpoint_path 参数，用于指定要加载的模型检查点路径
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default=None)
# 添加 --model_name 参数，用于指定保存模型时使用的名称
parser.add_argument('--model_name', type=str, default=None)
# 添加 --log_dir 参数，用于指定日志和TensorBoard文件的保存目录
parser.add_argument('--log_dir', default='logs/log')
# 添加 --num_point 参数，指定每个场景采样点的数量  md 这作者改别人的 default没对齐
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 20000]')
# 添加 --seed_feat_dim 参数，指定骨干网络输出的特征维度
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
# 添加 --voxel_size 参数，指定MinkowskiEngine处理点云时使用的体素大小
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size to process point clouds ')
# 添加 --max_epoch 参数，指定训练的总轮数
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 18]')
# 添加 --batch_size 参数，指定训练时的批量大小
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 2]')
# 添加 --learning_rate 参数，指定初始学习率
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
# 添加 --resume 参数，一个标志，如果使用此参数，则表示从检查点恢复训练
parser.add_argument('--resume', action='store_true', default=False, help='Whether to resume from checkpoint')
# 解析所有定义的命令行参数，并将结果存储在cfgs对象中
cfgs = parser.parse_args()



# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
# ------------------------------------------------------------------------- 全局配置设置



EPOCH_CNT = 0  #全局变量，用于跟踪当前训练周期
#设置全局变量，如检查点路径 CHECKPOINT_PATH。同时，配置日志系统，将训练信息打印到控制台并保存到 log_train.txt 文件中。
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None and cfgs.resume else None   #设置默认检查点路径，没有指定就使用默认路径
# 检查日志目录是否存在，如果不存在则创建
if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

#打开日志文件，以追加模式('a')写入
LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
#首先记录配置参数到日志
LOG_FOUT.write("配置参数："+str(cfgs) + '\n')

#定义log_string函数，同时将信息输出到日志文件和控制台
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n') # 写入日志文件
    LOG_FOUT.flush()               # 立即将缓冲区内容写入文件
    print(out_str)                 # 打印到控制台






# --- 2. 数据准备：初始化数据集和数据加载器 ---
# 为数据加载器中的每个工作进程设置不同的随机种子，确保数据增强的随机性
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


#调用此函数将所有物体的抓取标签一次性加载到内存中。
grasp_labels = load_grasp_labels(cfgs.dataset_root)
#实例化数据集。在这一步，数据集对象内部会构建好所有数据样本的文件路径列表。
TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, grasp_labels=grasp_labels, camera=cfgs.camera, split='train',
                                num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                remove_outlier=True, augment=True, load_label=True)
print('train dataset length训练样本总数: ', len(TRAIN_DATASET))  # 打印训练数据集的样本总数

#创建数据加载器。它会从 TRAIN_DATASET 中自动提取数据，并按照 batch_size（例如4）进行打包。
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
                              num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
print('train dataloader length训练数据加载器的批次总数: ', len(TRAIN_DATALOADER))# 打印训练数据加载器的批次总数





# --- 3. 模型构建：初始化模型和优化器 ---

# 实例化GraspNet模型，并设置为训练模式
print(f"检测到 {torch.cuda.device_count()} 张可用的GPU")
net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=True)
# 设置计算设备，如果CUDA可用则使用GPU，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 将模型的所有参数和缓冲区移动到指定的计算设备
net.to(device)
# Load the Adam optimizer
# 初始化Adam优化器，将模型的参数和学习率传递给它
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate)
# 初始化起始周期为0
start_epoch = 0

#如果 CHECKPOINT_PATH 有效，则加载之前保存的模型权重和优化器状态，实现断点续训。
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)                      # 加载检查点文件
    net.load_state_dict(checkpoint['model_state_dict'])           # 加载模型状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # 加载优化器状态
    start_epoch = checkpoint['epoch']  # 设置起始周期
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch)) # 记录日志
# TensorBoard Visualizers
# 初始化TensorBoard写入器，用于记录训练过程中的指标
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))

# 定义一个函数，根据当前周期计算学习率（指数衰减）
def get_current_lr(epoch):
    lr = cfgs.learning_rate    # 获取初始学习率
    lr = lr * (0.95 ** epoch)  # 每过一个周期，学习率乘以0.95
    return lr

# 定义一个函数，调整优化器中的学习率
def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)                  # 获取当前周期应有的学习率
    for param_group in optimizer.param_groups:  # 遍历优化器中的所有参数组
        param_group['lr'] = lr                   # 更新学习率



# --- 4. 训练循环 ---
# 定义单个周期的训练函数
#遍历 TRAIN_DATALOADER，每个迭代取出一个批次的数据 batch_data_label。
def train_one_epoch():
    stat_dict = {}                              # collect statistics  创建统计字典，用于收集训练指标
    adjust_learning_rate(optimizer, EPOCH_CNT)  ##根据当前全局周期调整学习率
    net.train()                                 ## 将模型设置为训练模式
    batch_interval = 20                         #定义一个日志记录间隔。这意味着每处理20个批次的数据，程序就会打印一次这段时间的平均训练指标。
     #遍历训练数据加载器的每个批次   我可以在这里设置看看，cpu和Gpu的内存变化
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER): #这是一个字典，包含了由 minkowski_collate_fn 整合好的、一个批次（例如4个场景）的所有数据。
        # 将批次数据中的所有张量移动到计算设备（GPU/CPU）
        for key in batch_data_label:
            if 'list' in key:  # 特殊处理列表嵌套的张量
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:               # 处理普通张量
                batch_data_label[key] = batch_data_label[key].to(device)

        # --- 模型训练的标准三步 ---
        # 1. 前向传播：将数据送入网络，得到包含预测结果的end_points字典
        end_points = net(batch_data_label)
        # 2. 计算损失：将end_points送入损失函数，得到总损失和各项子损失
        loss, end_points = get_loss(end_points)
        # 3. 反向传播和优化
        loss.backward()            # 根据总损失计算梯度
        optimizer.step()           # 优化器根据梯度更新模型参数
        optimizer.zero_grad()      # 清除梯度，为下一次迭代做准备

        # --- 收集和累加统计数据 ---
        # 遍历end_points字典，累加所有与损失和评估指标相关的键值
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict:  # 如果键不存在，则初始化为0
                    stat_dict[key] = 0    # 累加指标值
                stat_dict[key] += end_points[key].item()


        # --- 周期性地记录日志 ---
        # 判断是否达到了记录日志的时间点
        if (batch_idx + 1) % batch_interval == 0:
            # 打印周期和批次信息
            log_string(' ----epoch: %03d  ---- batch: %03d ----' % (EPOCH_CNT, batch_idx + 1))
            # 遍历已收集的指标
            for key in sorted(stat_dict.keys()):
                # 计算这个记录周期内的平均指标值 将平均指标值写入TensorBoard
                TRAIN_WRITER.add_scalar(key, stat_dict[key] / batch_interval,
                                                        
                                        (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size)
                # 将平均指标值打印到日志
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                # 清零，为下一个记录周期做准备
                stat_dict[key] = 0



# 定义主训练函数
def train(start_epoch):
    global EPOCH_CNT                                                       # 声明将要修改全局变量EPOCH_CNT
        # 从起始周期循环到最大周期
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch                                                  # 更新全局周期计数器
        log_string('**** EPOCH %03d ****' % epoch)                         # 记录新周期的开始
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))  # 记录当前学习率
        log_string(str(datetime.now()))                                    # 记录当前时间
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
          # 重置numpy的随机种子，这有助于增加不同epoch之间数据加载的随机性
        np.random.seed()
        train_one_epoch()                                                  # 调用函数，执行一整个周期的训练


        # --- 5. 日志与保存：保存模型检查点 ---
        # 创建一个字典，用于保存模型状态、优化器状态和周期数
        save_dict = {'epoch': epoch + 1, 'optimizer_state_dict': optimizer.state_dict(),
                     'model_state_dict': net.state_dict()}
        # 构建保存路径和文件名         # 保存检查点文件
        torch.save(save_dict, os.path.join(cfgs.log_dir, cfgs.model_name + '_epoch' + str(epoch + 1).zfill(2) + '.tar'))


if __name__ == '__main__':
    train(start_epoch)
