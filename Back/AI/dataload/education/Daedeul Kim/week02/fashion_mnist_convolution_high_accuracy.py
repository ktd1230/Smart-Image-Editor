import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import timeit


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # shape (?, 28, 28, 1)
        # conv  (?, 26, 26, 6)      (28-3)/1+1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)
        )

        # shape (?, 26, 26, 6)
        # conv  (?, 24, 24, 6)                (26-3)/1+1
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)
        )

        # shape (?, 24, 24, 6)
        # conv  (?, 22, 22, 12)                      (24-3)/1+1=22
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=0),
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )

        # shape (?, 22, 22, 12)
        # conv  (?, 20, 20, 12)                 (22-3)/1+1 = 20
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=0),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Linear(20 * 20 * 12, 120),
            nn.BatchNorm1d(120),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Linear(120, 60),
            nn.BatchNorm1d(60),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Linear(60, 10),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        return out


NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST("F_MNIST_data", train=True,
                                                 download=True,
                                                 transform=transform)
    testset = torchvision.datasets.FashionMNIST("F_MNIST_data", train=False,
                                                download=True,
                                                transform=transform)
    print(trainset.data.shape)
    print(testset.data.shape)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    costs = []
    total_batch = len(trainloader)
    start = timeit.default_timer()  # 수행시간 측정 시작
    for epoch in range(NUM_EPOCHS):
        total_cost = 0

        for i, (imgs, labels) in enumerate(trainloader):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)  # cost
            loss.backward()
            optimizer.step()

            total_cost += loss

        avg_cost = total_cost / total_batch
        print("Epoch:", "%03d" % (epoch + 1), "Cost =", "{:.9f}".format(avg_cost))
        costs.append(avg_cost)

    end = timeit.default_timer()  # 수행시간 측정 종료

    print("소요 시간 : {:.5f} sec".format((end - start)))
    print('%0.2f minutes' % ((end - start) / 60))

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

    label_tags = {
        0: 'T-Shirt',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle Boot'
    }

    columns = 6
    rows = 6
    fig = plt.figure(figsize=(10, 10))

    model.eval()
    for i in range(1, columns * rows + 1):
        data_idx = np.random.randint(len(testset))
        input_img = testset[data_idx][0].unsqueeze(dim=0).to(device)

        output = model(input_img)
        _, argmax = torch.max(output, 1)
        pred = label_tags[argmax.item()]
        label = label_tags[testset[data_idx][1]]

        fig.add_subplot(rows, columns, i)
        if pred == label:
            plt.title(pred + ', right !!')
            cmap = 'Blues'
        else:
            plt.title('Not ' + pred + ' but ' + label)
            cmap = 'Reds'
        plot_img = testset[data_idx][0][0, :, :]
        plt.imshow(plot_img, cmap=cmap)
        plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main()
