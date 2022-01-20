from operator import neg
import torch
import torch.nn as nn
import numpy as np

class ConfMSELoss(nn.Module):
    def __init__(self, reduction):
        super(ConfMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        pos_id = (targets==1.0).float()
        neg_id = (targets==0.0).float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss

def loss(pred_conf, pred_cls, pred_txtytwth, label):
    obj = 5.0
    noobj = 1.0

    # create loss functions
    conf_loss_function = ConfMSELoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduction='none') # bc of classification? ~~ probability
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none') # bc there's 2?
    twth_loss_function = nn.MSELoss(reduction='none') # hmmm

    pred_conf = torch.sigmoid(pred_conf[:, :, 0])
    pred_cls = pred_cls.permute(0, 2, 1)
    pred_txty = pred_txtytwth[:, :, :2]
    pred_twth = pred_txtytwth[:, :, 2:]

    gt_obj = label[:, :, 0].float()
    gt_cls = label[:, :, 1].long()
    gt_txtytwth = label[:, :, 2:-1].float()
    gt_box_scale_weight = label[:, :, -1]

    gt_txty = gt_txtytwth[:, :, :2]
    gt_twth = gt_txtytwth[:, :, 2:]

    # calculate loss
    # objectness loss
    pos_loss, neg_loss = conf_loss_function(pred_conf, gt_obj)
    conf_loss = obj * pos_loss + noobj * neg_loss

    # class loss
    cls_loss = torch.mean(torch.sum(cls_loss_function(pred_cls, gt_cls) * gt_obj, 1))

    # box loss
    txty_loss = torch.mean(torch.sum(torch.sum(txty_loss_function(pred_txty, gt_txty), 2) * gt_box_scale_weight * gt_obj, 1)) # why sum 2 times??
    twth_loss = torch.mean(torch.sum(torch.sum(twth_loss_function(pred_twth, gt_twth), 2) * gt_box_scale_weight * gt_obj, 1))
    txtytwth_loss = txty_loss + twth_loss

    total_loss = conf_loss + cls_loss + txtytwth_loss

    return conf_loss, cls_loss, txtytwth_loss, total_loss

def generate_dxdywh(gt_label, w, h, s):
    xmin, xmax, ymin, ymax = gt_label[:-1]
    # compute center, width and height
    c_x = (xmin + xmax) / 2 * w
    c_y = (ymin + ymax) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    if box_w < 1. or box_h < 1.:
        return False

    # map center point of box to the grid cell
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s) # specific grid (bound) have to be an int?
    grid_y = int(c_y_s)

    # compute (x, y, w, h) for the corresponding grid cell
    tx = c_x_s - grid_x # offset to the top left corner
    ty = c_y_s - grid_y
    tw = np.log(box_w) # offset from cluster centroids
    th = np.log(box_h)
    weight = 2.0 - (box_w / w) * (box_h / h) # to balance the loss between large and small objects

    return grid_x, grid_y, tx, ty, tw, th, weight

def gt_creator(input_size, stride, label_lists=[]):
    assert len(input_size) > 0 and len(label_lists) > 0
    # prepare all empty gt data
    batch_size = len(label_lists)
    w = input_size[1]
    h = input_size[0]

    # make gt labels by anchor-free method and anchor-based method (??)
    ws = w // stride
    hs = h // stride
    s = stride
    gt_tensor = np.zeros([batch_size, hs, ws, 1+1+4+1])

    # generate gt whose style is yolov1
    for batch_idx in range(batch_size):
        for gt_label in label_lists[batch_idx]:
            gt_class = gt_label[-1]
            result = generate_dxdywh(gt_label, w, h, s)
            if result:
                grid_x, grid_y, tx, ty, tw, th, weight = result

                if grid_x < gt_tensor.shape[2] and grid_y < gt_tensor.shape[1]:
                    gt_tensor[batch_idx, grid_y, grid_x, 0] = 1.0
                    gt_tensor[batch_idx, grid_y, grid_x, 1] = gt_class
                    gt_tensor[batch_idx, grid_y, grid_x, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[batch_idx, grid_y, grid_x, 6] = weight
    
    gt_tensor = gt_tensor.reshape(batch_size, -1, 1+1+4+1)

    return gt_tensor