""" Loss functions for training.
    Author: chenxi-wang
"""

import torch
import torch.nn as nn


def get_loss(end_points):
    objectness_loss, end_points = compute_objectness_loss(end_points)
    graspness_loss, end_points = compute_graspness_loss(end_points)
    view_loss, end_points = compute_view_graspness_loss(end_points)
    score_loss, end_points = compute_score_loss(end_points)
    width_loss, end_points = compute_width_loss(end_points)
    loss = objectness_loss + 10 * graspness_loss + 100 * view_loss + 15 * score_loss + 10 * width_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_objectness_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    sa2_inds = end_points['sa2_inds'].long()
    objectness_label = torch.gather(objectness_label, 1, sa2_inds)
    loss = criterion(objectness_score, objectness_label)
    end_points['loss/stage1_objectness_loss'] = loss

    # objectness_pred = torch.argmax(objectness_score, 1)
    # end_points['stage1_objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()
    # end_points['stage1_objectness_prec'] = (objectness_pred == objectness_label.long())[
    #     objectness_pred == 1].float().mean()
    # end_points['stage1_objectness_recall'] = (objectness_pred == objectness_label.long())[
    #     objectness_label == 1].float().mean()
    return loss, end_points


def compute_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    graspness_score = end_points['graspness_score'].squeeze(1)
    graspness_label = end_points['graspness_label'].squeeze(-1)
    
    sa2_inds = end_points['sa2_inds'].long()
    graspness_label = torch.gather(graspness_label, 1, sa2_inds)
    loss_mask = end_points['objectness_label'].bool()
    loss_mask = torch.gather(loss_mask, 1, sa2_inds)
    
    loss = criterion(graspness_score, graspness_label)
    loss = loss[loss_mask]
    loss = loss.mean()
    
    # graspness_score_c = graspness_score.detach().clone()[loss_mask]
    # graspness_label_c = graspness_label.detach().clone()[loss_mask]
    # graspness_score_c = torch.clamp(graspness_score_c, 0., 0.99)
    # graspness_label_c = torch.clamp(graspness_label_c, 0., 0.99)
    # rank_error = (torch.abs(torch.trunc(graspness_score_c * 20) - torch.trunc(graspness_label_c * 20)) / 20.).mean()
    # end_points['stage1_graspness_acc_rank_error'] = rank_error

    # graspness_score_c[graspness_score_c > 0.15] = 1
    # graspness_score_c[graspness_score_c <= 0.15] = 0
    # graspness_label_c[graspness_label_c > 0.15] = 1
    # graspness_label_c[graspness_label_c <= 0.15] = 0
    # end_points['graspness_acc'] = (graspness_score_c.bool() == graspness_label_c.bool()).float().mean()

    end_points['loss/stage1_graspness_loss'] = loss

    return loss, end_points


def compute_view_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_graspness']
    loss = criterion(view_score, view_label)
    end_points['loss/stage2_view_loss'] = loss
    return loss, end_points


def compute_score_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    grasp_score_pred = end_points['grasp_score_pred']
    grasp_score_label = end_points['batch_grasp_score']
    loss = criterion(grasp_score_pred, grasp_score_label)
    # loss_mask = end_points['objectness_label'].bool()
    # loss_mask = torch.gather(loss_mask, 1, end_points['inds_graspable'])
    # loss = loss[loss_mask].mean()
    end_points['loss/stage3_score_loss'] = loss
    # end_points['loss/stage3_score_loss_point_num'] = loss_mask.sum() / B
    return loss, end_points


def compute_width_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_width_pred = end_points['grasp_width_pred']
    # norm by cylinder radius(Crop module) and norm to 0~1, original with is 0~0.1, /0.05->0~2, /2->0~1
    grasp_width_label = end_points['batch_grasp_width'] * 10   # norm by cylinder radius(Crop module)
    loss = criterion(grasp_width_pred, grasp_width_label)
    grasp_score_label = end_points['batch_grasp_score']
    loss_mask = grasp_score_label > 0
    loss = loss[loss_mask].mean()
    end_points['loss/stage3_width_loss'] = loss
    return loss, end_points
