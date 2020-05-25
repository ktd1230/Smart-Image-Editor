import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.FashionMNIST('../../../datasets/education/', download=True, train=True, transform=transform)

testset = datasets.FashionMNIST('../../../datasets/education/', download=True, train=False, transform=transform)

# 1 channel 28 width 28 height
mnist_train = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
mnist_test = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True)
# nn 패키지를 사용하고 모델과 손실 함수를 정의합니다.
input_size = 28*28
H = 1000
output_size = 10

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.log_softmax(self.l4(x), dim=1)
        return x


model = Model()
model.to(device)
# optimizer 설정 Adam, SGD
# optimizer = optim.SGD(model.parameters(), lr = 0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# loss
criterion = torch.nn.CrossEntropyLoss()

# hyper-parameters

epoch = 30
num_batches = len(mnist_train)

train_losses = []
test_losses = []

for e in range(epoch):
    train_loss = 0
    test_loss = 0
    accuracy = 0
    for images, labels in mnist_train:
        images = images.to(device)
        labels = labels.to(device)
        result = model(images)
        loss = criterion(result, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
    print("Epoch: {}/{}.. ".format(e + 1, epoch),
          "Training Loss: {:.3f}.. ".format(train_loss / len(mnist_train)),
          "Test Loss: {:.3f}.. ".format(test_loss / len(mnist_test)),
          "Test Accuracy: {:.3f}".format(accuracy / len(mnist_test)))
    train_losses.append(train_loss / len(mnist_train))
    test_losses.append(test_loss / len(mnist_test))


# Rescale 샘플 크기를 조절하는 클래스이지만 지금 데이터는 28*28로 일정하므로 할 필요가 없다.
# class Rescale(object):
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size
#
#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks']
#         h, w = image.shape[:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size
#         new_h, new_w = int(new_h), int(new_w)
#         img = transform.resize(image, (new_h, new_w))
#         landmarks = landmarks * [new_w / w, new_h / h]
#         return {'image': img, 'landmarks': landmarks}

