import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(1)

data = np.load("../../../../datasets/education/linear_train.npy")


x, y = zip(*data)
nx = np.asarray(x)
ny = np.asarray(y)
nx = np.expand_dims(nx, axis=1)
ny = np.expand_dims(ny, axis=1)

x_train = torch.FloatTensor(nx)
y_train = torch.FloatTensor(ny)

# 입력의 차원, 출력의 차원을 인수로 받는다.
model = nn.Linear(1, 1)

print(list(model.parameters()))

optimizer = torch.optim.SGD(model.parameters(), lr=0.004)

nb_epochs = 2000
for epoch in range(nb_epochs+1):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

torch.save(model.state_dict(), "../../checkpoints/model.pth")

xdata = np.load("../../../datasets/education/linear_test_x.npy")

xdata = np.expand_dims(xdata, axis=1)
test_data = torch.FloatTensor(xdata)
pred_y = model(test_data)

