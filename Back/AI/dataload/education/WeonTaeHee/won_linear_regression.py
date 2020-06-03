import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# 난수 발생 순서와 값을 동일하게 보장해준다는 특징
torch.manual_seed(1)

data = np.load("../../../../datasets/education/linear_train.npy")


x, y = zip(*data)
nx = np.asarray(x)
ny = np.asarray(y)
x_train = torch.FloatTensor(nx)
y_train = torch.FloatTensor(ny)

# 선형 회귀란 학습 데이터와 가장 잘 맞는 하나의 직선을 찾는 일
# 가장 잘 맞는직선을 정의하는 것은 W와 b값을 찾는것
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.004)

nb_epochs = 2000
for epoch in range(nb_epochs + 1):

    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train) ** 2)
    # gradient를 0으로 초기화
    optimizer.zero_grad()

    # 비용 함수를 미분하여 gradient 계산
    cost.backward()

    #W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}' .format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
c 