import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

trainset = datasets.FashionMNIST('../../datasets/education/', download=True, train=True, transform=transform)
mnist_train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.FashionMNIST('../../datasets/education/', download=True, train=False, transform=transform)
mnist_test = torch.utils.data.DataLoader(testset, batch_size=64,shuffle=True)

# 1 channel 28 width 28 height

print(trainset.data.shape)
print(testset.data.shape)

# linear model 작성할 것..
