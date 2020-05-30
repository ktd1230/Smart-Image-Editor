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



