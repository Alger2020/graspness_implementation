import torch.nn as nn
import torch

'''
文件核心功能
loss.py 的核心功能是定义和计算 GraspNet 模型在训练过程中所需的所有损失函数。
GraspNet 是一个多阶段、多任务的模型，它同时预测物体性、抓取度、抓取视角、抓取分数和抓取宽度。
因此，它的总损失是这五个子任务损失的加权和。这个文件精确地定义了每个子任务的损失如何计算，
以及它们如何组合成最终的总损失，用于驱动模型的反向传播和参数更新。
'''


'''功能: 这是整个损失计算的入口函数。
 它按顺序调用其他五个独立的损失计算函数，然后根据预设的权重（1, 10, 100, 15, 10）
 将它们组合成一个单一的标量值 loss。
 权重的作用: 不同的权重反映了不同任务的重要性。例如，view_loss 的权重高达 100，表明让模型学会选择正确的抓取接近方向是至关重要的。
'''
def get_loss(end_points):
    # --- 分别计算五个子任务的损失 ---
    
    # 1. 计算物体性损失 (分类任务)
    objectness_loss, end_points = compute_objectness_loss(end_points)
    # 2. 计算抓取度损失 (回归任务)
    graspness_loss, end_points = compute_graspness_loss(end_points)
    # 3. 计算视角抓取度损失 (回归任务)
    view_loss, end_points = compute_view_graspness_loss(end_points)
    # 4. 计算最终抓取分数损失 (回归任务)
    score_loss, end_points = compute_score_loss(end_points)
    # 5. 计算抓取宽度损失 (回归任务)
    width_loss, end_points = compute_width_loss(end_points)
    # --- 将所有损失加权求和，得到总损失 ---  有趣，这个比例是怎么确定的
    loss = objectness_loss + 10 * graspness_loss + 100 * view_loss + 15 * score_loss + 10 * width_loss
    
    # 将总损失存入end_points字典，用于记录    
    end_points['loss/overall_loss'] = loss
    # 返回总损失和更新后的end_points字典
    return loss, end_points



'''目标: 监督 GraspableNet 模块，使其能正确区分点云中的物体点和背景点。
方法: 这是一个二分类问题，因此使用交叉熵损失 (nn.CrossEntropyLoss)。
额外指标: 除了损失，还计算了准确率（acc）、精确率（prec）和召回率（recall），这些指标能更直观地反映模型在分类任务上的表现'''
def compute_objectness_loss(end_points):
    # 使用交叉熵损失，这是分类任务的标准损失函数
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # 提取模型预测的物体性分数 (B, 2, N) 和真值标签 (B, N)
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    # 计算损失
    loss = criterion(objectness_score, objectness_label)
    # 将损失存入字典
    end_points['loss/stage1_objectness_loss'] = loss

    # --- 计算并记录评估指标 (准确率、精确率、召回率) ---
    objectness_pred = torch.argmax(objectness_score, 1)   # 获取预测类别 (0或1)
    end_points['stage1_objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()
    end_points['stage1_objectness_prec'] = (objectness_pred == objectness_label.long())[
        objectness_pred == 1].float().mean()
    end_points['stage1_objectness_recall'] = (objectness_pred == objectness_label.long())[
        objectness_label == 1].float().mean()
    return loss, end_points



'''
目标: 监督 GraspableNet 模块，使其能准确预测每个物体点的“可抓取”程度（一个0到1之间的连续值）。
方法: 这是一个回归问题，使用Smooth L1 损失。
关键点: loss_mask 的使用非常重要。
它确保了只在被正确分类为“物体”的点上计算抓取度损失，避免了让模型去学习背景点的抓取度，这既没有意义也可能干扰训练。
'''
def compute_graspness_loss(end_points):
    # 使用Smooth L1损失，这是一种对异常值不那么敏感的回归损失
    criterion = nn.SmoothL1Loss(reduction='none')
    # 提取模型预测的抓取度分数和真值标签 (B,1,N) （B,N,1)
    graspness_score = end_points['graspness_score'].squeeze(1)
    graspness_label = end_points['graspness_label'].squeeze(-1)
    # 创建一个掩码，只对物体点计算损失
    loss_mask = end_points['objectness_label'].bool()
    # 计算逐点的损失
    loss = criterion(graspness_score, graspness_label)
    # 应用掩码，只保留物体点的损失
    loss = loss[loss_mask]
    # 对有效损失求平均
    loss = loss.mean()
    
    
    # ... (计算 rank_error 指标) ...
    graspness_score_c = graspness_score.detach().clone()[loss_mask]
    graspness_label_c = graspness_label.detach().clone()[loss_mask]
    graspness_score_c = torch.clamp(graspness_score_c, 0., 0.99)
    graspness_label_c = torch.clamp(graspness_label_c, 0., 0.99)
    rank_error = (torch.abs(torch.trunc(graspness_score_c * 20) - torch.trunc(graspness_label_c * 20)) / 20.).mean()
    end_points['stage1_graspness_acc_rank_error'] = rank_error
    end_points['loss/stage1_graspness_loss'] = loss
    return loss, end_points


'''
目标: 监督 ApproachNet 模块。
对于每个抓取候选点，ApproachNet 会为300个预设的接近方向（视角）打分。此损失函数的目标就是让这些分数尽可能地接近真值。
方法: 同样是回归问题，使用 Smooth L1 损失。
'''
def compute_view_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    # 提取模型预测的每个视角的抓取度和真值标签
    view_score = end_points['view_score'] #(B,Ns1024,num_view300) 
    view_label = end_points['batch_grasp_view_graspness'] #(B,Ns,V)
    loss = criterion(view_score, view_label)
    # 计算损失
    end_points['loss/stage2_view_loss'] = loss
    return loss, end_points




'''
目标: 这两个函数都用于监督 SWADNet 模块，即模型的最后精调阶段。
compute_score_loss: 让模型学会为每个具体的抓取姿态（由视角、旋转角、深度共同定义）预测一个准确的质量分数。
compute_width_loss: 让模型学会为每个姿态预测正确的抓取器张开宽度。
方法: 都是回归问题，使用 Smooth L1 损失。
关键点: 在 compute_width_loss 中，同样使用了掩码 loss_mask。
它确保只在那些真值分数大于0（即有效的、成功的抓取）的姿态上计算宽度损失。预测一个失败抓取的宽度是没有意义的
'''
def compute_score_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    # 提取最终预测的抓取分数和真值
    grasp_score_pred = end_points['grasp_score_pred'] #(B, Ns, num_angle, num_depth)
    grasp_score_label = end_points['batch_grasp_score']  # (B, Ns, V, A, D) 
    loss = criterion(grasp_score_pred, grasp_score_label)

    end_points['loss/stage3_score_loss'] = loss
    return loss, end_points



def compute_width_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    # 提取最终预测的抓取宽度和真值
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_label = end_points['batch_grasp_width'] * 10
    loss = criterion(grasp_width_pred, grasp_width_label)
    # 只在有效的抓取上计算宽度损失
    grasp_score_label = end_points['batch_grasp_score']
    loss_mask = grasp_score_label > 0
    loss = loss[loss_mask].mean()
    end_points['loss/stage3_width_loss'] = loss
    return loss, end_points
