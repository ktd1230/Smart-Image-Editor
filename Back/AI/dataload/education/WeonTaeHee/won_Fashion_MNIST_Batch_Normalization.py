import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
                                          torch.nn.BatchNorm2d(32), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                                          torch.nn.BatchNorm2d(64), torch.nn.ReLU())
        self.fc = torch.nn.Linear(24 * 24 * 64, 256, bias=True)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seed_all(777)
learning_rate = 0.001
training_epochs = 30
batch_size = 256

trainset = datasets.FashionMNIST('../../../datasets/education/', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('../../../datasets/education/', download=True, train=False, transform=transform)

mnist_train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
mnist_test = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
model = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(mnist_train)
print("총 배치의 수 : {}".format(total_batch))

train_losses = []
test_losses = []
for epoch in range(training_epochs):
    avg_cost = 0
    train_loss = 0
    test_loss = 0
    accuracy = 0
    for X, Y in mnist_train:
        X = X.to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        loss = cost
        cost.backward()
        optimizer.step()
        avg_cost += cost / total_batch
        train_loss += loss.item()
    else:
        with torch.no_grad():
            model.eval()
            for images, labels in mnist_test:
                images = images.to(device)
                labels = labels.to(device)
                log_ps = model(images)
                prob = torch.exp(log_ps)
                top_probs, top_classes = prob.topk(1, dim=1)
                equals = labels == top_classes.view(labels.shape)
                accuracy += equals.type(torch.FloatTensor).mean()
                test_loss += criterion(log_ps, labels)
        model.train()
    print("Epoch: {}/{}.. ".format(epoch + 1, training_epochs),
          "Training Loss: {:.3f}.. ".format(train_loss / len(mnist_train)),
          "Test Loss: {:.3f}.. ".format(test_loss / len(mnist_test)),
          "Test Accuracy: {:.3f}".format(accuracy / len(mnist_test)))
    train_losses.append(train_loss / len(mnist_train))
    test_losses.append(test_loss / len(mnist_test))