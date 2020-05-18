import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import timeit


class FashionMNISTLinear(nn.Module):
    def __init__(self):
        super(FashionMNISTLinear, self).__init__()

        # 2-Dim : 28 x 28
        # 1-Dim : 784
        # Layer 3
        # self.layer1 = nn.Sequential(
        #     torch.nn.Linear(784, 256, bias=True),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.ReLU()
        # )
        #
        # self.layer2 = nn.Sequential(
        #     torch.nn.Linear(256, 64, bias=True),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.ReLU()
        # )
        #
        # self.layer3 = nn.Sequential(
        #     torch.nn.Linear(64, 8, bias=True)
        # )

        # Layer 4
        # self.layer1 = nn.Sequential(
        #     torch.nn.Linear(784, 256, bias=True),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.ReLU()
        # )
        #
        # self.layer2 = nn.Sequential(
        #     torch.nn.Linear(256, 128, bias=True),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.ReLU()
        # )
        #
        # self.layer3 = nn.Sequential(
        #     torch.nn.Linear(128, 64, bias=True),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.ReLU()
        # )
        #
        # self.layer4 = nn.Sequential(
        #     torch.nn.Linear(64, 8, bias=True)
        # )

        # # Layer 5
        self.layer1 = nn.Sequential(
            torch.nn.Linear(784, 256, bias=True),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            torch.nn.Linear(256, 128, bias=True),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            torch.nn.Linear(128, 64, bias=True),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            torch.nn.Linear(64, 32, bias=True),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            torch.nn.Linear(32, 10, bias=True)
        )

        # Layer 6
        # self.layer1 = nn.Sequential(
        #     torch.nn.Linear(784, 256, bias=True),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.ReLU()
        # )
        #
        # self.layer2 = nn.Sequential(
        #     torch.nn.Linear(256, 128, bias=True),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.ReLU()
        # )
        #
        # self.layer3 = nn.Sequential(
        #     torch.nn.Linear(128, 64, bias=True),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.ReLU()
        # )
        #
        # self.layer4 = nn.Sequential(
        #     torch.nn.Linear(64, 32, bias=True),
        #     torch.nn.BatchNorm1d(32),
        #     torch.nn.ReLU()
        # )
        #
        # self.layer5 = nn.Sequential(
        #     torch.nn.Linear(32, 16, bias=True),
        #     torch.nn.BatchNorm1d(16),
        #     torch.nn.ReLU()
        # )
        #
        # self.layer6 = nn.Sequential(
        #     torch.nn.Linear(16, 8, bias=True)
        # )

    def forward(self, x):
        # Layer 3
        # x = x.view(x.size(0), -1)  # flatten
        # x_out = self.layer1(x)
        # x_out = self.layer2(x_out)
        # x_out = self.layer3(x_out)
        # return x_out

        # Layer 4
        # x = x.view(x.size(0), -1)  # flatten
        # x_out = self.layer1(x)
        # x_out = self.layer2(x_out)
        # x_out = self.layer3(x_out)
        # x_out = self.layer4(x_out)
        # return x_out

        # Layer 5
        x = x.view(x.size(0), -1)  # flatten
        x_out = self.layer1(x)
        x_out = self.layer2(x_out)
        x_out = self.layer3(x_out)
        x_out = self.layer4(x_out)
        x_out = self.layer5(x_out)
        return x_out

        # Layer 6
        # x = x.view(x.size(0), -1)  # flatten
        # x_out = self.layer1(x)
        # x_out = self.layer2(x_out)
        # x_out = self.layer3(x_out)
        # x_out = self.layer4(x_out)
        # x_out = self.layer5(x_out)
        # x_out = self.layer6(x_out)
        # return x_out


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


def main():
    batch_size = 100
    num_epochs = 20
    learning_rate = 0.0001
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
    train_data = dset.FashionMNIST("F_MNIST_data", train=True, transform=transform, download=True)
    test_data = dset.FashionMNIST("F_MNIST_data", train=False, transform=transform, download=True)

    print(train_data.data.shape)
    print(test_data.data.shape)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = FashionMNISTLinear().to(device)
    model.apply(weights_init)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    costs = []
    total_batch = len(train_loader)
    start = timeit.default_timer()  # 수행시간 측정 시작
    for epoch in range(num_epochs):
        total_cost = 0

        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)  # cost

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_cost += loss

        avg_cost = total_cost / total_batch
        print("Epoch:", "%03d" % (epoch + 1), "Cost =", "{:.9f}".format(avg_cost))
        costs.append(avg_cost)

    end = timeit.default_timer()  # 수행시간 측정 종료

    print("소요 시간 : {:.5f} sec".format((end-start)))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, argmax = torch.max(outputs, 1)
            total += imgs.size(0)
            correct += (labels == argmax).sum().item()

        print('Accuracy for {} images: {:.2f}%'.format(total, correct / total * 100))

    # 이미지 분류 시각화
    # 참고한 자료 : http://www.gisdeveloper.co.kr/?p=7846
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
        data_idx = np.random.randint(len(test_data))
        input_img = test_data[data_idx][0].unsqueeze(dim=0).to(device)

        output = model(input_img)
        _, argmax = torch.max(output, 1)
        pred = label_tags[argmax.item()]
        label = label_tags[test_data[data_idx][1]]

        fig.add_subplot(rows, columns, i)
        if pred == label:
            plt.title(pred + ', right !!')
            cmap = 'Blues'
        else:
            plt.title('Not ' + pred + ' but ' + label)
            cmap = 'Reds'
        plot_img = test_data[data_idx][0][0, :, :]
        plt.imshow(plot_img, cmap=cmap)
        plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main()
