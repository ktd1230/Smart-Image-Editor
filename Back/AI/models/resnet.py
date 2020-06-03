import torch
import math
import torch.nn as nn
import numpy as np


class Bottleneck(nn.Module):
    def __init__(self,inplane,planes,stride=1,downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplane,planes[0],kernel_size=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.conv2 = nn.Conv2d(planes[0],planes[1],kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.conv3 = nn.Conv2d(planes[1],planes[2],kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes[2])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            out += self.downsample(x)

        out = self.bn3(out)
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self,layers,cfg=None,type="Bottleneck"):
        super(ResNet,self).__init__()
        if cfg == None:
            cfg = [[64],[64,64,256]*layers[0],[128,128,512]*layers[1],[256,256,1024]*layers[2],[512,512,2048]*layers[3]]
        self.cfg = cfg
        inplane = cfg[0][0]
        self.initial_layer = self.initial_network(inplane)
        # self.conv1 = nn.Conv2d(3,self.inplanes,kernel_size=7,stride=2,padding=3,bias=False)
        # self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        block = Bottleneck
        self.layer1 = self._make_layer(block,inplane,cfg[1],layers[0],1)
        inplane = cfg[1][3*layers[0]-1]
        self.layer2 = self._make_layer(block, inplane, cfg[2], layers[1], 2)
        inplane = cfg[2][3 * layers[1] - 1]
        self.layer3 = self._make_layer(block, inplane, cfg[3], layers[2], 3)
        inplane = cfg[3][3 * layers[2] - 1]
        self.layer4 = self._make_layer(block, inplane, cfg[4], layers[3], 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()


    def initial_network(self,inplane):
        entrance_net = nn.Sequential(
            nn.Conv2d(3,inplane,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        return entrance_net

    def _make_layer(self,block,inplane,cfg,number_of_layers,target_layer):
        layers = []
        for i in range(number_of_layers):
            downsample = None
            stride = 1
            if i == 0 :
                stride = 2
                if target_layer == 1:
                    stride = 1
                downsample = nn.Conv2d(inplane,cfg[2],kernel_size=1,stride=stride,bias=False)
            layers.append(block(inplane,cfg[i*3:i*3+3],stride=stride,downsample=downsample))
            inplane = cfg[i*3+2]
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.initial_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out




if __name__  == "__main__":
    cfg = [[32],[27,24,10,5,4,2,1,9,2],[23,23,12,4,6,1,2,6,1],[9,2,3,5,19,2,5,9,10],[1,2,3,4,5,6,7,8,9]]
    resnet = ResNet([3,4,23,3])
    print(resnet)
    x = torch.rand(1,3,224,224)
    out = resnet(x)
    print(out)