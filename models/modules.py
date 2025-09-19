import os
import sys
from networkx import jaccard_coefficient
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import pointnet2.pytorch_utils as pt_utils
from pointnet2.pointnet2_utils import CylinderQueryAndGroup
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix


class GraspableNet(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv_graspable = nn.Conv1d(self.in_dim, 3, 1)

    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(seed_features)  # (B, 3, num_seed)
        end_points['objectness_score'] = graspable_score[:, :2]
        end_points['graspness_score'] = graspable_score[:, 2]
        return end_points


'''ApproachNet 模块核心功能
ApproachNet 的核心任务是为每一个经过筛选的抓取候选点（seed point）预测一个最佳的抓取接近方向。
想象一下，对于桌上的一个苹果，你可以从上方抓，也可以从侧面抓。ApproachNet 就是要分析这个点的特征，
然后从一系列预设的接近方向（比如300个覆盖半球的向量）中，为这个点挑选出最合适的一个。
'''
class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim, is_training=True):
        super().__init__()
        self.num_view = num_view  # 预设的接近方向(视角)总数，例如300
        self.in_dim = seed_feature_dim  # 输入的每个点的特征维度，例如512
        self.is_training = is_training # 训练/推理标志
        # 定义两个1D卷积层，它们相当于作用于每个点特征的全连接层
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)

    def forward(self, seed_features, end_points):
        # 从输入特征张量 (B, C, num_seed) 中获取批次大小B和抓取候选点数量num_seed
        B, _, num_seed = seed_features.size()
        # 将输入特征通过第一个1D卷积层和ReLU激活函数，提取更深层的特征
        # res_features的形状仍为 (B, C, num_seed)，它将作为残差特征返回
        res_features = F.relu(self.conv1(seed_features), inplace=True)
        # 将中间特征送入第二个1D卷积层，将特征维度从C映射到num_view
        # 输出 features 的形状为 (B, num_view, num_seed)，代表每个点在每个预设视角上的得分
        features = self.conv2(res_features)
        # 转置最后两个维度，得到形状 (B, num_seed, num_view)，这样更直观：每个点都有一组视角分数
        view_score = features.transpose(1, 2).contiguous() # (B, num_seed, num_view)
        # 将原始的视角分数存入 end_points 字典，用于后续的损失计算
        end_points['view_score'] = view_score

        if self.is_training:
            # normalize view graspness score to 0~1
            # 【训练模式】：进行概率采样，增加探索性,      
            # 复制一份分数张量并分离计算图，避免影响反向传播     
            view_score_ = view_score.clone().detach()
            # 沿着视角维度(dim=2)，找到每个点的最高分
            view_score_max, _ = torch.max(view_score_, dim=2)
            # 沿着视角维度(dim=2)，找到每个点的最低分
            view_score_min, _ = torch.min(view_score_, dim=2)
            # 将最高分张量扩展到与原始分数张量相同的形状 (B, num_seed, num_view)
            view_score_max = view_score_max.unsqueeze(-1).expand(-1, -1, self.num_view)
            # 将最低分张量扩展到与原始分数张量相同的形状 (B, num_seed, num_view)
            view_score_min = view_score_min.unsqueeze(-1).expand(-1, -1, self.num_view)
            # 对分数进行min-max归一化到[0, 1]区间，作为采样的概率分布。加1e-8防止除零
            view_score_ = (view_score_ - view_score_min) / (view_score_max - view_score_min + 1e-8)

            # 初始化一个列表，用于收集每个批次样本采样到的视角索引
            top_view_inds = []
            # 遍历批次中的每一个场景
            for i in range(B):
                # 根据归一化后的分数(概率)，为当前场景的每个点随机采样一个视角索引
                #假设你有一个骰子，但每个面出现的概率不一样。比如，6点那一面特别重，
                #所以掷出6点的概率比其他点数都高。torch.multinomial 就是用来模拟这种根据指定概率进行抽样的过程。
                top_view_inds_batch = torch.multinomial(view_score_[i], 1, replacement=False)
                # 将采样结果添加到列表中
                top_view_inds.append(top_view_inds_batch)
                
            # 将列表中的张量堆叠起来，并移除多余的维度，得到形状 (B, num_seed)
            top_view_inds = torch.stack(top_view_inds, dim=0).squeeze(-1)  # B, num_seed
        else:
            # 【推理模式】：直接选择分数最高的视角
            # 沿着视角维度(dim=2)，找到分数最高视角的索引
            _, top_view_inds = torch.max(view_score, dim=2)  # (B, num_seed)
           
            # --- 根据选出的最佳视角索引，计算对应的旋转矩阵 ---
            # 扩展索引张量的维度，方便后续使用torch.gather进行索引
            top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
            # 生成预设的、覆盖半球的 num_view 个视角向量 (num_view, 3)
            template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)
            # 扩展预设视角张量，使其能与每个点的索引对应
            template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()
            # 使用gather函数，根据top_view_inds_为每个点选出对应的视角向量(xyz坐标)
            vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
            # 将视角向量展平，为计算旋转矩阵做准备
            vp_xyz_ = vp_xyz.view(-1, 3)
            # 创建一个全零的角度张量，因为这里只关心接近方向，不考虑绕轴的旋转
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
            # 将视角向量(作为Z轴)和0度角转换为3x3旋转矩阵
            vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
            # 将计算出的最佳视角向量存入end_points
            
            end_points['grasp_top_view_xyz'] = vp_xyz  #(B,num_seed,3)
            # 将计算出的最佳视角旋转矩阵存入end_points
            end_points['grasp_top_view_rot'] = vp_rot  #(B,num_seed,3,3)
            
        # 将最终选出的视角索引(无论是采样还是argmax得到)存入end_points
        end_points['grasp_top_view_inds'] = top_view_inds
        # 返回包含预测结果的end_points字典和第一层卷积输出的残差特征
        return end_points, res_features




'''
CloudCrop 模块核心功能
CloudCrop 的核心任务是为每一个抓取候选点，提取其周围的局部几何特征。
可以把它想象成一个可定向的“数字放大镜”。在 ApproachNet 确定了最佳的接近方向（即放大镜的朝向）后，CloudCrop 就将这个放大镜对准抓取候选点，
"裁剪"出该点周围一个圆柱体区域内的所有点云，并分析这个小区域内的点云几何结构，最终将其提炼成一个更具描述性的特征向量。
这个过程至关重要，因为它使得网络不再仅仅依赖于单个点的特征，
而是能够理解抓取点周围的局部形状（例如，这个点是在一个平面上、一个边缘上还是一个角落里），这对于后续精确预测抓取宽度和分数至关重要。'''
class CloudCrop(nn.Module):
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.04):
        super().__init__()
        self.nsample = nsample # 在圆柱体内采样的邻居点数量
        self.in_dim = seed_feature_dim  # 输入特征维度
        self.cylinder_radius = cylinder_radius # 圆柱体半径
        mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3 # MLP的维度，3代表xyz坐标

        # 关键组件：圆柱体查询与分组
        self.grouper = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
                                             use_xyz=True, normalize_xyz=True)
        # 用于处理分组后特征的共享MLP (mini-PointNet)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        # 1. 分组：为每个抓取点裁剪出一个圆柱形局部点云  # (B, C+3, Ns, K)
        grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, vp_rot,
                                       seed_features_graspable)  # B*3 + feat_dim*M*K
        # 2. 特征提取：用MLP处理每个局部点云的特征    # (B, 256, Ns , K)
        new_features = self.mlps(grouped_feature)  # (batch_size, mlps[-1], M, K)
        # 3. 聚合：对每个局部点云的特征进行最大池化，得到一个聚合特征  # (B, 256, Ns , 1)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (batch_size, mlps[-1], M, 1)
        # 4. 降维：移除多余的维度   # (B, 256, Ns )
        new_features = new_features.squeeze(-1)   # (batch_size, mlps[-1], M)
        return new_features




'''SWADNet 模块核心功能
SWADNet 是整个抓取检测网络的最终预测头。它的名字是 Score, Width, Angle, Depth 的缩写，
精准地概括了它的功能：对于每一个抓取候选点，在已经确定的最佳接近方向下，
预测所有可能的抓取器旋转角度（Angle）和滑动深度（Depth）组合的抓取分数（Score）和抓取宽度（Width）。
可以把它理解为抓取姿态的“精调”和“评估”阶段。
'''
class SWADNet(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle  # 抓取器旋转角度的数量，例如 12
        self.num_depth = num_depth  # 抓取器滑动深度的数量，例如 4
        # 第一个卷积层，用于特征变换，输入维度必须与CloudCrop的输出维度一致
        self.conv1 = nn.Conv1d(256, 256, 1)  # input feat dim need to be consistent with CloudCrop module
        # 关键的预测层，将特征映射到所有预测值上
        #这是核心的预测层。它非常巧妙地将每个点的256维特征，一次性地映射为 2 * num_angle * num_depth 个值。
        #这里的 2 代表了两个预测目标：抓取分数和抓取宽度。
        self.conv_swad = nn.Conv1d(256, 2*num_angle*num_depth, 1) 

    def forward(self, vp_features, end_points):
        # 获取输入特征的形状信息
        B, _, num_seed = vp_features.size()
        # 步骤1: 特征提取
        vp_features = F.relu(self.conv1(vp_features), inplace=True)
        # 步骤2: 核心预测
        vp_features = self.conv_swad(vp_features)
        # 步骤3: 维度重组 (Reshape)
        vp_features = vp_features.view(B, 2, self.num_angle, self.num_depth, num_seed)
        # 步骤4: 维度重排 (Permute)
        vp_features = vp_features.permute(0, 1, 4, 2, 3)

        # split prediction
        # 步骤5: 分离预测结果
        end_points['grasp_score_pred'] = vp_features[:, 0]  # (B, Ns, num_angle, num_depth) 即所有点的抓取分数预测。
        end_points['grasp_width_pred'] = vp_features[:, 1] # (B, Ns, num_angle, num_depth) 即所有点的抓取宽度预测。
        return end_points
