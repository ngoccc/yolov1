import numpy as np
import torch
import torch.nn as nn
from backbone import resnet18
from utils import Conv, SPP, SAM, BottleneckCSP
import tools

class MyYOLO(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=True, conf_thresh=0.01, nms_thresh=0.5):
        super(MyYOLO, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = 32

        self.grid_cell = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]]) # ???
        self.scale_torch = torch.tensor(self.scale.copy(), device=device).float()
        
        # backbone: ResNet18
        self.backbone = resnet18(pretrained=True)

        # neck: SPP & SAM
        self.SPP = nn.Sequential(
            Conv(512, 256, k=1),
            SPP(),
            BottleneckCSP(256*4, 512, n=1, shortcut=False)
        )
        self.SAM = SAM(512)
        self.conv_set = BottleneckCSP(512, 512, n=3, shortcut=False)

        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)
    
         
    def create_grid(self, input_size):
        w, h = input_size[1], input_size[0]
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 2).to(self.device)
        
        return grid_xy

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()

    def forward(self, x, target=None):
        # back bone
        C5 = self.backbone(x)

        # head
        C5 = self.SPP(C5)
        C5 = self.SAM(C5)
        C5 = self.conv_set(C5)

        # pred - fc
        pred = self.pred(C5)
        pred = pred.view(C5.size(0), 1 + self.num_classes + 4, -1).permute(0, 2, 1)
        B, HW, C = pred.size()

        # divide prediction to obj_pred, txtytwth_pred and cls_pred
        conf_pred = pred[:, :, :1]
        cls_pred = pred[:, :, 1 : 1 + self.num_classes]
        txtytwth_pred = pred[:, :, 1 + self.num_classes:]

        if self.trainable: # train
            conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(
                pred_conf=conf_pred, pred_cls=cls_pred, pred_txtytwth=txtytwth_pred, label=target)
            return conf_loss, cls_loss, txtytwth_loss, total_loss
        else: # test
            pass