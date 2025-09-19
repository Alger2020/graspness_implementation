""" GraspNet baseline model definition.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from models.backbone_resunet14 import MinkUNet14D
from models.modules import ApproachNet, GraspableNet, CloudCrop, SWADNet
from loss_utils import GRASP_MAX_WIDTH, NUM_VIEW, NUM_ANGLE, NUM_DEPTH, GRASPNESS_THRESHOLD, M_POINT
from label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation


'''GraspNet 模块整体概述
GraspNet 是一个端到端的抓取检测模型。它的核心思想是分阶段、由粗到精地预测抓取。整个流程可以概括为：
1.场景理解：首先，使用一个强大的3D骨干网络（MinkUNet14D）从整个输入点云中提取每个点的深层特征。
2.抓取点定位：然后，通过 GraspableNet 模块预测每个点的“物体性”（是否在物体上）和“抓取度”（是否适合抓取），从而筛选出少量高质量的抓取候选点（seed points）。
3.抓取姿态估计：
    姿态粗估计：ApproachNet 模块为这些候选点预测最佳的抓取接近方向（approach vector）。
    局部区域裁剪：CloudCrop 模块在每个候选点周围，沿着预测出的最佳接近方向，裁剪出一个圆柱形的局部点云区域。
    姿态精调与评估：最后，SWADNet 模块分析这个局部区域，对抓取姿态进行微调，并预测最终的抓取宽度、深度和质量分数。

'''
class GraspNet(nn.Module):
    def __init__(self, cylinder_radius=0.05, seed_feat_dim=512, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = NUM_DEPTH    # 抓取深度的离散数量
        self.num_angle = NUM_ANGLE    # 抓取器旋转角度的离散数量
        self.M_points = M_POINT       # 经过最远点采样后保留的抓取候选点数量
        self.num_view = NUM_VIEW      # 抓取接近方向(视角)的离散数量

        #  骨干网络，用于提取点云的逐点特征
        #  一个基于稀疏卷积的3D U-Net。它负责处理输入的稀疏体素化点云，并输出每个点的高维特征。  输入3，输出512,D是告知为3D数据。
        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
       
       
        #  预测“物体性”和“抓取度”的网络
        #  一个小型的网络，接收骨干网络输出的特征，并预测两个东西：每个点属于物体的概率（物体性），以及每个点适合抓取的程度（抓取度）。
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        #  预测抓取接近方向（视角）的网络
        # 接收抓取候选点的特征，并预测出一组最佳的抓取接近方向。
        self.rotation = ApproachNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        # 裁剪局部点云的模块
        # 一个非学习型模块，根据给定的抓取候选点和接近方向，在原始场景中裁剪出圆柱形区域的点云和特征。
        self.crop = CloudCrop(nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)
        # 预测抓取宽度、角度、深度的网络
        # 接收裁剪后的局部点云特征，并最终预测出精确的抓取参数（宽度、深度、分数）。
        self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth)

    def forward(self, end_points):
        seed_xyz = end_points['point_clouds']  # use all sampled point cloud, B*Ns*3
        B, point_num, _ = seed_xyz.shape  # batch _size
        
        
        # 点云特征提取模块
        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']
        #将普通的点云数据（坐标和特征）打包成 MinkowskiEngine 专门使用的、高效的“稀疏张量”格式
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch) 
        # 经过排序和重塑后，最终得到的 seed_features 是一个形状为 (B, 512, N) 的张量
        seed_features = self.backbone(mink_input).F
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)  #输出（B,512,20000）
        #特征提取模块为什么不筛选出可抓取点后再进行特征提取


        # 抓取点预测与筛选 
        # 将特征送入GraspableNet，预测物体性和抓取度，结果存入end_points 
        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)             #  # 将特征维度从(B, C, N)转置为(B, N, C)，方便后续按点索引
        objectness_score = end_points['objectness_score']                 # 从end_points中取出物体性预测分数 (B,N,2)
        graspness_score = end_points['graspness_score'].squeeze(1)        # 从end_points中取出抓取度预测分数，并移除多余的维度
        objectness_pred = torch.argmax(objectness_score, 1)               # 沿类别维度找最大值的索引，得到每个点的物体性预测(0:背景, 1:物体)
        objectness_mask = (objectness_pred == 1)                          # 创建物体性掩码：预测为物体的点为True
        graspness_mask = graspness_score > GRASPNESS_THRESHOLD            # 创建抓取度掩码：抓取度分数高于阈值的点为True
        graspable_mask = objectness_mask & graspness_mask                 # 合并两个掩码，得到最终的“可抓取”点掩码


        #最远点采样(FPS) 
        # 初始化空列表，用于收集每个批次样本处理后的结果
        seed_features_graspable = []
        seed_xyz_graspable = []
        # 初始化一个计数器，用于统计整个批次中可抓取点的总数
        graspable_num_batch = 0.
        # 遍历批次中的每一个场景
        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim
            cur_seed_xyz = seed_xyz[i][cur_mask]  # Ns*3

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # 1*Ns*3
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*Ns
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # Ns*3
            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1*feat_dim*Ns
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous() # feat_dim*Ns

            seed_features_graspable.append(cur_feat)  
            seed_xyz_graspable.append(cur_seed_xyz)
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # B*Ns*3 # 形状: (B, M, 3)
        seed_features_graspable = torch.stack(seed_features_graspable)  # B*feat_dim*Ns  # 形状: (B, C, M)
        
        end_points['xyz_graspable'] = seed_xyz_graspable # 将采样后的点坐标存入end_points
        end_points['graspable_count_stage1'] = graspable_num_batch / B    # 计算并存储平均每个场景的可抓取点数量


        # 接近方向预测 (ApproachNet) 
        # 将采样后的特征送入ApproachNet，预测接近方向，并得到残差特征
        end_points, res_feat = self.rotation(seed_features_graspable, end_points)  #res_features #（B，C512,Ns1024)
        # 将残差特征加回到原始特征上，增强特征表达
        seed_features_graspable = seed_features_graspable + res_feat


        # 局部区域裁剪 (CloudCrop) 
        # 如果是训练模式，需要匹配真值标签来确定用于裁剪的旋转矩阵
        if self.is_training:
            # 准备抓取标签
            end_points = process_grasp_labels(end_points)
            # 将预测的视角与真值标签进行匹配，得到用于监督的旋转矩阵
            grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points)
            
            # 如果是推理模式，直接使用ApproachNet预测出的最佳视角旋转矩阵
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']
        # 调用CloudCrop模块，裁剪出M个圆柱形局部区域的特征
        group_features = self.crop(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
       
       
       
       
       
         # 抓取姿态精调 (SWADNet) 
        # 将局部区域特征送入SWADNet，预测最终的抓取分数和宽度
        end_points = self.swad(group_features, end_points)
        # 返回包含所有预测结果和中间变量的字典
        return end_points




'''pred_decode 核心功能
pred_decode 函数的作用是解码器。它在推理（inference）阶段被调用，负责将 GraspNet 模型输出的
、高度离散化和多维度的原始预测张量（raw predictions），转换成一系列具体的、易于使用的抓取姿态参数。
简单来说，它的任务是回答以下问题：
对于每个抓取候选点，最佳的抓取角度和深度组合是什么？
这个最佳组合的抓取质量分数是多少？
对应的抓取宽度、角度、深度、旋转矩阵和中心点坐标分别是多少？'''
def pred_decode(end_points):
        # 获取批次大小，通常在推理时为 1
    batch_size = len(end_points['point_clouds'])
        # 初始化一个空列表，用于存放每个场景的最终抓取预测结果
    grasp_preds = []
        # 遍历批次中的每个场景
    for i in range(batch_size):
        # 提取当前场景的 M 个抓取候选点的坐标
        grasp_center = end_points['xyz_graspable'][i].float() # 形状: (M, NUM_ANGLE, NUM_DEPTH) -> (1024, 12, 4)
        # 提取 SWADNet 输出的原始抓取分数预测
        grasp_score = end_points['grasp_score_pred'][i].float() 
        # 将最后两个维度（角度和深度）展平，便于寻找最大值
        grasp_score = grasp_score.view(M_POINT, NUM_ANGLE*NUM_DEPTH)  # 形状: (1024, 48)
        # 对每个候选点，在48个(角度,深度)组合中找到分数最高的那个，并返回分数和对应的索引
        grasp_score, grasp_score_inds = torch.max(grasp_score, -1)  # [M_POINT]# grasp_score形状(1024,), inds形状(1024,)
        # 将最高分整理成列向量
        grasp_score = grasp_score.view(-1, 1) # 形状: (1024, 1)
        
        # --- 根据最高分索引解码出角度和深度 ---
        # 使用整数除法，从展平的索引中恢复出角度索引 (0-11)
        # 再乘以每个角度的步长 (pi/12)，得到弧度制的抓取器平面内旋转角
        grasp_angle = (grasp_score_inds // NUM_DEPTH) * np.pi / 12
        
        # 使用取模运算，从展平的索引中恢复出深度索引 (0-3)
        # 索引+1再乘以每个深度的步长(0.01m)，得到米为单位的抓取深度
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1) * 0.01
        # 将深度整理成列向量
        grasp_depth = grasp_depth.view(-1, 1)
        # 提取 SWADNet 输出的原始宽度预测，并进行缩放
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.  # 形状: (1024, 12, 4)
        # 同样将宽度预测展平
        grasp_width = grasp_width.view(M_POINT, NUM_ANGLE*NUM_DEPTH)  # 形状: (1024, 48)
         # 使用之前找到的最高分索引(grasp_score_inds)，从48个宽度预测中精确地挑出与最佳(角度,深度)组合对应的那个宽度
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))  # 形状: (1024, 1)
        # 将抓取宽度限制在合理的范围内 [0, GRASP_MAX_WIDTH]
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)

        # 提取 ApproachNet 预测的最佳接近方向向量 (approaching vector)
        approaching = -end_points['grasp_top_view_xyz'][i].float() # 形状: (1024, 3)
        # 调用辅助函数，将接近方向向量和平面内旋转角，转换为一个完整的3x3旋转矩阵
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)  # 形状: (1024, 3, 3)
        # 将3x3的旋转矩阵展平为9维向量，便于拼接
        grasp_rot = grasp_rot.view(M_POINT, 9) # 形状: (1024, 9)



        # merge preds
        # --- 合并所有预测结果 ---
        # 定义一个固定的抓取高度（夹爪厚度的一半）
        grasp_height = 0.02 * torch.ones_like(grasp_score) # 形状: (1024, 1)
        # 定义物体ID，这里不预测物体ID，所以设为-1
        obj_ids = -1 * torch.ones_like(grasp_score)  # 形状: (1024, 1)
        
                
        # 按照GraspNet API要求的顺序，将所有参数拼接成一个大的张量
        # 顺序: [score, width, height, depth, rotation (9), center (3), obj_id]
        # 最终形状: (1024, 1+1+1+1+9+3+1) -> (1024, 17)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
                
    # 返回一个列表，其中每个元素是对应场景的所有抓取预测
    return grasp_preds
