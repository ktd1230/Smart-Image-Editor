import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        try:
            train = np.load("../../../datasets/education/linear_train.npy")
        except Exception as e:
            print(e)

        self.x_train = train[:, [0]]
        self.x_train = torch.FloatTensor(self.x_train)
        print(self.x_train[:5], self.x_train.shape)

        self.y_train = train[:, [1]]
        self.y_train = torch.FloatTensor(self.y_train)
        print(self.y_train[:5], self.y_train.shape)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_train[idx])
        y = torch.FloatTensor(self.y_train[idx])
        return x, y


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 100
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples

        prediction = model(x_train)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx + 1, len(dataloader), cost.item()
        ))


new_var = torch.FloatTensor([[73], [80], [75]])
pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75일 때의 예측값 : ", pred_y)