import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# device = 'cpu'

# 기본경로
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))
# print(BASE_DIR)

learning_rate = 0.001
training_epochs = 15
batch_size = 64

# transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = dsets.FashionMNIST(BASE_DIR + "\datasets\education", download=True, train=True, transform=transform)
testset = dsets.FashionMNIST(BASE_DIR + "\datasets\education", download=True, train=False, transform=transform)

# 2) data loader 생성 및 Batch 사이즈 결정(보통 Batch사이즈는 2의 배수로 잡음 - gpu의 하드웨어 자원  특성 고려...)
# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1).to(device),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1).to(device),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(7 * 7 * 64, 10, bias=True).to(device)
        )

    def forward(self, x):
        x_out = self.layer1(x)
        x_out = self.layer2(x_out)
        x_out = x_out.view(x_out.size(0), -1)
        x_out = self.layer3(x_out)
        return x_out

model = DNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    costs = []
    total_batch = len(trainloader)
    for epoch in range(training_epochs):
        total_cost = 0

        for i, (imgs, labels) in enumerate(trainloader):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_cost += loss

        avg_cost = total_cost / total_batch
        print("Epoch:", "%03d" % (epoch + 1), "Cost =", "{:.9f}".format(avg_cost))
        costs.append(avg_cost)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (imgs, labels) in enumerate(testloader):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, argmax = torch.max(outputs, 1)
                total += imgs.size(0)
                correct += (labels == argmax).sum().item()

            print('Accuracy for {} images: {:.2f}%'.format(total, correct / total * 100))


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
