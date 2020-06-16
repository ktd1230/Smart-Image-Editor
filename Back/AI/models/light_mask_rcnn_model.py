import math
import sys
import time
import torch
from torchvision import models,utils
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone,BackboneWithFPN
import torch.nn as nn
from models.resnet import ResNet


def get_mask_rcnn_model(layers,num_classes,out_channels=256,cfg=None):
    backbone = ResNet(layers,cfg)
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    in_channels_list = []
    cfg = backbone.cfg
    for i in range(1, len(cfg)):
        layer_size = len(cfg[i])
        in_channels_list.append(cfg[i][layer_size - 1])
    print(in_channels_list)
    backbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
    model = models.detection.MaskRCNN(backbone, num_classes)
    return model

def get_pruned_config(custom_mrcnn_model,percent):
    total_size = 0
    bn = []
    for m in custom_mrcnn_model.parameters():
        if isinstance(m,nn.BatchNorm2d):
            bn.append(m)
            total_size += m.weight.data.shape[0]

    candidate_for_bn = torch.zeros(total_size)
    index = 0
    for m in bn:
        size = m.weight.data.shape[0]
        candidate_for_bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size
    y,i = torch.sort(candidate_for_bn)
    threshold_index = int(total_size * percent)
    threshold = y[threshold_index]

    pruned = 0
    pruned_cfg = []
    for m in bn:
        weight_copy = m.weight.data.clone()
        mask = weight_copy.abs().gt(threshold).float()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        pruned_cfg.append(int(torch.sum(mask)))

    before_cfg = custom_mrcnn_model.cfg
    new_cfg = []
    index = 0
    for cfg_list in before_cfg:
        size = len(cfg_list)
        new_cfg.append([ele for ele in pruned_cfg[index:index+size]])
        index +=size
    return new_cfg





