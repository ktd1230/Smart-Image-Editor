import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import math

import numpy as np

def feature_test():
    vgg = models.vgg16(pretrained=True)
    #print(vgg.features)
    for layer, (name, module) in enumerate(vgg.features._modules.items()):
        print("layer",layer)
        print("name", name)
        print("module", module)

def grad_test():
    x = torch.ones(2,2,requires_grad=True)
    loss_fn = torch.nn.MSELoss()
    target = torch.tensor([[0.,1.],[2.,3.]])
    output = loss_fn(x,target)
    output.backward()
    optimizer = optim.SGD([x],lr=1)
    print("x",x)
    print("x.grad",x.grad)
    optimizer.step()
    x.grad.data.add_(torch.tensor([[0.1,0.1],[0.1,0.1]]))
    print("x.grad", x.grad)
    print("after step",x)
    batchnorm1  = nn.BatchNorm2d(32)
    for i in range(16):
        batchnorm1.weight.data[i] = 0.
    batchnorm2 = nn.BatchNorm2d(16)
    for i in range(16):
        batchnorm2.weight.data[i] = 0.5
    batchnorm3 = nn.BatchNorm2d(8)
    total_size = 56
    bn = torch.zeros(total_size)
    m_list = [batchnorm1,batchnorm2,batchnorm3]
    index = 0
    for m in m_list:
        size = m.weight.data.shape[0]
        bn[index : (index+size)] = m.weight.data.abs().clone()
        index += size
    print(bn)
    y,i = torch.sort(bn)
    thre_index = int(total_size*0.4)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k,m in enumerate(m_list):
        weight_copy = m.weight.data.clone()
        mask = weight_copy.abs().gt(thre).float()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
              format(k, mask.shape[0], int(torch.sum(mask))))




    print(batchnorm1)

def hook_test():
    torch.manual_seed(777)
    input = torch.rand(size=(1,1,6,6))


    layer = nn.Sequential(
        nn.Conv2d(1,8,kernel_size=2),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8,4,kernel_size=2),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Conv2d(4,2,kernel_size=2),
        nn.BatchNorm2d(2),
        nn.ReLU()
    )
    score = nn.Linear(2*3*3,1)
    output = layer(input)
    output = output.view(output.shape[0],-1)
    output = score(output)
    loss_fn = torch.nn.MSELoss()
    target = torch.tensor([1.])
    loss = loss_fn(output, target)

    loss.backward()
    for i,m in enumerate(layer.modules()):
        if isinstance(m,nn.BatchNorm2d):
            print("i 번째 batchnrom grad",m.weight.grad.data)

    for m in layer.modules():
        if isinstance(m,nn.BatchNorm2d):
            m.weight.grad.data.add_(torch.ones(m.weight.shape[0]))

    print("=====================================")
    for i,m in enumerate(layer.modules()):
        if isinstance(m,nn.BatchNorm2d):
            print("i 번째 batchnrom grad",m.weight.grad.data)

    param_list = list(layer.parameters()) + list(score.parameters())
    #optimizer = optim.SGD(param_list, lr=1)

def mask_rcnn_resnet_test():
    model_vanila = models.resnet50(pretrained=True)
    model_fpn = models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    model_fpn.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    prediction = model_fpn(x)
    print("original renset",model_vanila)
    print("======================================")
    print("fpn renset", model_fpn)

class vgg(nn.Module):

    def __init__(self, dataset='cifar10', init_weights=True, cfg=None):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'cifar10':
            num_classes = 10
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def load_model_test():
    model = vgg()
    torch.save(model.state_dict(),"C:\\Users\\multicampus\\s02p31c101\\Back\\AI\\dummy.pth")
    model.load_state_dict(torch.load("C:\\Users\\multicampus\\s02p31c101\\Back\\AI\\dummy.pth"))
    print(model)

def resnet101():
    resnet = models.resnet101(pretrained=True)
    print(resnet)
    #x = torch.rand(1,3, 300, 400)
    #out = resnet(x)
    #print(out)






grad_test()
#feature_test()
#hook_test()
#mask_rcnn_resnet_test()

#load_model_test()
#resnet101()

#preresnet_test()