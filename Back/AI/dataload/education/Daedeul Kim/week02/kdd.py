# from __future__ import print_function
# import torch
# x = torch.rand(5, 3)
# print(x)

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt


def display_result(criterion, model, x, y):
    pred = model(x)
    loss = criterion(input=pred, target=y)

    plt.clf()
    # plt.xlim(0, 11)
    # plt.ylim(0, 8)
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pred.data.numpy(), "b--")
    plt.title("loss={:.4}, w={:.4}, b={:.4}".format(loss.data.item(), model.weight.data.item(), model.bias.data.item()))
    plt.show()


def model_save(model, model_name):
    torch.save(obj=model, f=model_name)


def model_load(model_name):
    return torch.load(model_name)


def linear_regression_by_pytorch():
    # 1. 기본 설정
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    # 현재 실습(함수)에 대해 파이썬 코드를 재실행해도 같은 결과가 나오도록 랜덤 시드 구성
    torch.manual_seed(1)

    # 2. 변수 선언
    x_train = torch.FloatTensor([[1], [2], [3]])
    y_train = torch.FloatTensor([[2], [4], [6]])

    print(x_train)
    print(x_train.shape)  # x_train 크기가 3 x 1

    print(y_train)
    print(y_train.shape)  # y_train 크기가 3 x 1

    # 3. 가중치와 편향의 초기화
    """
    선형 회귀란 학습 데이터와 가장 잘 맞는 하나의 직선을 찾는 일이다.
    그리고 가장 잘 맞는 직성을 정의하는 것은 바로 W와 b이다.
    선형 회귀의 목표는 가장 잘 맞는 직선을 정의하는 W와 b의 값을 찾는 것이다.
    """

    # 가중치 W를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시한다.
    W = torch.zeros(1, requires_grad=True)

    # 가중치 W 출력
    print(W)

    b = torch.zeros(1, requires_grad=True)
    print(b)

    # 이 지점에서 직선의 방정식은, y=0*x+0
    # x에 어떤 값이 들어가더라도 가설은 0을 예측하게 된다.

    # 4. 가설 세우기
    # H(x) = Wx + b
    hypothesis = x_train * W + b
    print("hypothesis : {}".format(hypothesis))

    # 5. 비용 함수 선언하기
    cost = torch.mean((hypothesis - y_train) ** 2)
    print("cost : {}".format(cost))

    # 6. 경사 하강법 구현하기
    # SGD는 경사 하강법의 일종으로, lr은 학습률(learning rate)를 의미함.
    optimizer = optim.SGD([W, b], lr=0.01)

    optimizer.zero_grad()  # gradient를 0으로 초기화
    cost.backward()  # 비용 함수를 미분하여 gradient 계산
    optimizer.step()  # W와 b를 업데이터

    # Epoch(에포크)는 전체 훈련 데이터가 학습에 한 번 사용된 주기를 의미합니다.
    # 이 실습에서는 2000번 수행됐습니다.
    nb_epochs = 2000  # 원하는만큼 경사 하강법을 반복
    for epoch in range(nb_epochs + 1):

        # H(x) 계산
        hypothesis = x_train * W + b

        # cost 계산
        cost = torch.mean((hypothesis - y_train) ** 2)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 100회 반복마다 로그 출력
        if epoch % 100 == 0:
            print("Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}".format(
                epoch, nb_epochs, W.item(), b.item(), cost.item()
            ))

    # 훈련 결과를 보면, 최적의 기울기 W는 2에 가깝고 b는 0에 가까운데 (실제값 W:1.997, b:0.006)
    # x_train이 [[1], [2], [3]], y_train이 [[2], [4], [6]]인 것을 고려하면
    # 실제 정답은 W가 2이고 b가 0인 H(x) = 2x이므로 거의 정답을 찾은 것이다.

    # 5. 4에서 optimizer.zero_grad()가 필요한 이유
    """
    import torch
    w = torch.tensor(2.0, requires_grad=True)

    nb_epochs = 20
    for epoch in range(nb_epochs + 1):

        z = 2 * w

        z.backward()
        print("수식을 w로 미분한 값 : {}".format(w.grad))

    ===== result =====
    수식을 w로 미분한 값 : 2.0
    수식을 w로 미분한 값 : 4.0
    ...
    수식을 w로 미분한 값 : 42.0
    ==================

    계속해서 미분값인 2가 누적되기 때문에 optimizer.zero_grad()를 통해 미분값을 계속 0으로 초기화 한다.
    """


def process_1():
    data = pd.read_csv("02_Linear_Regression_Model_Data.csv")

    x = torch.from_numpy(data['x'].values).unsqueeze(dim=1).float()
    y = torch.from_numpy(data['y'].values).unsqueeze(dim=1).float()

    plt.xlim(0, 11)
    plt.ylim(0, 8)
    plt.scatter(x, y)

    plt.show()

    model = nn.Linear(in_features=1, out_features=1, bias=True)
    print(model)
    print(model.weight)
    print(model.bias)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    torch.optim.SGD

    print(model(x))

    for step in range(1000):
        pred = model(x)
        loss = criterion(input=pred, target=y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            pass

    display_result(criterion, model, x, y)
    # model_save(model, "02_Linear_Regression_Model.pt")


def process_2():
    try:
        test_x = np.load("../../../datasets/education/linear_test_x.npy")
        train = np.load("../../../datasets/education/linear_train.npy")
    except Exception as e:
        print(e)
        return

    x_train = train[:, [0]]
    x_train = torch.FloatTensor(x_train)
    print(x_train.shape)

    y_train = train[:, [1]]
    y_train = torch.FloatTensor(y_train)
    print(y_train.shape)

    W = torch.zeros(1, requires_grad=True)
    print(W)

    b = torch.zeros(1, requires_grad=True)
    print(b)

    # 4. 가설 세우기
    # H(x) = Wx + b
    hypothesis = x_train * W + b
    print("hypothesis : {}".format(hypothesis))

    # 5. 비용 함수 선언하기
    cost = torch.mean((hypothesis - y_train) ** 2)
    print("cost : {}".format(cost))

    # 6. 경사 하강법
    import torch.optim as optim

    # 학습률
    # 0.01 : Over shooting
    # 0.007 : Epoch 2000/2000 W: 15.389, b: 7.833 Cost: 900.938232
    # 0.005 : Epoch 2000/2000 W: 15.385, b: 7.780 Cost: 900.939209
    # 0.001 : Epoch 2000/2000 W: 15.141, b: 4.536 Cost: 903.672729
    # 0.0005 : Epoch 2000/2000 W: 14.981, b: 2.403 Cost: 908.336182
    # 0.0001 : Epoch 2000/2000 W: 14.781, b: -0.257 Cost: 917.338013
    # 0.00001 : Epoch 2000/2000 W: 14.657, b: -1.011 Cost: 921.170776
    optimizer = optim.SGD([W, b], lr=0.007)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    nb_epochs = 2000
    for epoch in range(nb_epochs + 1):

        hypothesis = x_train * W + b

        cost = torch.mean((hypothesis - y_train) ** 2)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print("Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}".format(
                epoch, nb_epochs, W.item(), b.item(), cost.item()
            ))

    # H(x) = 14.65x-1
    print(W, b)

    import torch.nn as nn
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()

    import torch.nn.functional as F

    print(list(model.parameters()))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.007)
    # Learning Rate Scheduler
    # 학습이 진행되면서 학습률을 상황에 맞게 변경시킬 수 있다면 더 낮은 loss값을 얻을 수 있다.
    # 이를 위해서는 학습률 스케쥴이 필요하고, 관련 코드는 아래와 같다.

    # 지정한 스텝 단위로 학습률에 감마를 곱해서 학습률을 감소시키는 방식
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    # 지정한 스텝 지점마다 학습률에 감마를 곱해서 감소시키는 방식
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 400, 1000], gamma=0.1)

    # 매 epoch마다 학습률에 감마를 곱해서 감소시키는 방식
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # 원하는 epoch마다, 이전 학습률 대비 변경폭에 따라 학습률을 감소시키는 방식
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1, patience=1, mode="min")

    """
    # 의사코드임에 유의한다.
    # StepLR, MultiStepLR, ExponentialLR 사용방법
    for i in range(epochs):
        scheduler.step() # 

        optimizer.zero_grad()
        output = model.forward(...)
        loss = loss(...)
        loss.backward()
        optimizer.step()
    
    # ReduceLROnPlateau 사용방법
    for i in range(epochs):
        optimizer.zero_grad()
        output = model.forward(...)
        loss = loss(...)
        loss.backward()
        optimizer.step()

        scheduler.step()  # 
    """

    nb_epochs = 2000
    for epoch in range(nb_epochs + 1):

        pred = model(x_train)
        cost = F.mse_loss(pred, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            #
            print("Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}".format(
                epoch, nb_epochs, W.item(), b.item(), cost.item()
            ))

    # model_save(model, "linear_train_model.pt")
    display_result(criterion, model, x_train, y_train)
    print(x_train[:5])
    print(y_train[:5])

    # for elem in test_x:
    #     new_var = torch.FloatTensor([elem])
    #     pred_y = model(new_var)
    #     print(pred_y)

    new_x = torch.FloatTensor([test_x]).t()
    print(new_x[:5])
    pred_y = model(new_x)
    print(test_x[:5])
    print(pred_y[:5])
    display_result(criterion, model, new_x, pred_y)


def process_3_by_model():
    try:
        test_x = np.load("../../../datasets/education/linear_test_x.npy")
        model = model_load("linear_train_model.pt")
        print(" model: {} \n type: {}".format(model, type(model)))
    except Exception as e:
        print(e)
        return

    import torch.nn as nn

    criterion = nn.MSELoss()
    new_x = torch.FloatTensor([test_x]).t()
    pred_y = model(new_x)
    display_result(criterion, model, new_x, pred_y)
    pass


def learning_rate_scheduler():
    try:
        train = np.load("../../../datasets/education/linear_train.npy")
    except Exception as e:
        print(e)
        return

    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F

    x_train = train[:, [0]]
    x_train = torch.FloatTensor(x_train)
    print(x_train.shape)

    y_train = train[:, [1]]
    y_train = torch.FloatTensor(y_train)
    print(y_train.shape)

    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    print(list(model.parameters()))

    # 학습률
    # 0.01 : Over shooting
    # 0.007 : Epoch 2000/2000 W: 15.389, b: 7.833 Cost: 900.938232
    # 0.005 : Epoch 2000/2000 W: 15.385, b: 7.780 Cost: 900.939209
    # 0.001 : Epoch 2000/2000 W: 15.141, b: 4.536 Cost: 903.672729
    # 0.0005 : Epoch 2000/2000 W: 14.981, b: 2.403 Cost: 908.336182
    # 0.0001 : Epoch 2000/2000 W: 14.781, b: -0.257 Cost: 917.338013
    # 0.00001 : Epoch 2000/2000 W: 14.657, b: -1.011 Cost: 921.170776
    optimizer = torch.optim.SGD(model.parameters(), lr=0.007)

    # Learning Rate Scheduler
    # 학습이 진행되면서 학습률을 상황에 맞게 변경시킬 수 있다면 더 낮은 loss값을 얻을 수 있다.
    # 이를 위해서는 학습률 스케쥴이 필요하고, 관련 코드는 아래와 같다.

    # 지정한 스텝 단위로 학습률에 감마를 곱해서 학습률을 감소시키는 방식
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    # 지정한 스텝 지점마다 학습률에 감마를 곱해서 감소시키는 방식
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 400, 1000], gamma=0.1)

    # 매 epoch마다 학습률에 감마를 곱해서 감소시키는 방식
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # 원하는 epoch마다, 이전 학습률 대비 변경폭에 따라 학습률을 감소시키는 방식
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1, patience=1, mode="min")

    """
    # 의사코드임에 유의한다.
    # StepLR, MultiStepLR, ExponentialLR 사용방법
    for i in range(epochs):
        optimizer.zero_grad()
        output = model.forward(...)
        loss = loss(...)
        loss.backward()
        optimizer.step()
        
        scheduler.step() # 

    # ReduceLROnPlateau 사용방법
    for i in range(epochs):
        scheduler.step()  # 
        
        optimizer.zero_grad()
        output = model.forward(...)
        loss = loss(...)
        loss.backward()
        optimizer.step()
    """

    nb_epochs = 2000
    for epoch in range(nb_epochs + 1):

        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        scheduler.step()
        # scheduler.step(cost)  # ReduceLROnPlateau

        if epoch % 100 == 0:
            #
            print("Epoch {:4d}/{} Cost: {:.6f}".format(
                epoch, nb_epochs, cost.item()
            ))

    # model_save(model, "linear_train_model.pt")
    display_result(criterion, model, x_train, y_train)
    print(x_train[:5])
    print(y_train[:5])


def use_gpu_device():
    try:
        train = np.load("../../../datasets/education/linear_train.npy")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(e)
        return

    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F

    x_train = train[:, [0]]
    x_train = torch.cuda.FloatTensor(x_train)
    print(x_train[:5])
    print(x_train.shape)

    y_train = train[:, [1]]
    y_train = torch.cuda.FloatTensor(y_train)
    print(y_train[:5])
    print(y_train.shape)

    model = nn.Linear(1, 1)
    model = nn.DataParallel(model)
    criterion = nn.MSELoss().cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.007)

    # Learning Rate Scheduler
    # 학습이 진행되면서 학습률을 상황에 맞게 변경시킬 수 있다면 더 낮은 loss값을 얻을 수 있다.
    # 이를 위해서는 학습률 스케쥴이 필요하고, 관련 코드는 아래와 같다.

    # 지정한 스텝 단위로 학습률에 감마를 곱해서 학습률을 감소시키는 방식
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    # 지정한 스텝 지점마다 학습률에 감마를 곱해서 감소시키는 방식
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 400, 1000], gamma=0.1)

    # 매 epoch마다 학습률에 감마를 곱해서 감소시키는 방식
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # 원하는 epoch마다, 이전 학습률 대비 변경폭에 따라 학습률을 감소시키는 방식
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1, patience=1, mode="min")

    """
    # 의사코드임에 유의한다.
    # StepLR, MultiStepLR, ExponentialLR 사용방법
    for i in range(epochs):
        optimizer.zero_grad()
        output = model.forward(...)
        loss = loss(...)
        loss.backward()
        optimizer.step()

        scheduler.step() # 

    # ReduceLROnPlateau 사용방법
    for i in range(epochs):
        scheduler.step()  # 

        optimizer.zero_grad()
        output = model.forward(...)
        loss = loss(...)
        loss.backward()
        optimizer.step()
    """

    nb_epochs = 10000
    for epoch in range(nb_epochs + 1):

        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        scheduler.step()
        # scheduler.step(cost)  # ReduceLROnPlateau

        if epoch % 100 == 0:
            #
            print("Epoch {:4d}/{} Cost: {:.6f}".format(
                epoch, nb_epochs, cost.item()
            ))

    # model_save(model, "linear_train_model.pt")

    print(x_train[:5])
    print(y_train[:5])


def main():
    # linear_regression_by_pytorch()
    # process_1()
    # process_2()
    # process_3_by_model()
    # learning_rate_scheduler()
    use_gpu_device()
    return


if __name__ == "__main__":
    main()
