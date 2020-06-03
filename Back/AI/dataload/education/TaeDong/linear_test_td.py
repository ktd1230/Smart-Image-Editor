import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
print(BASE_DIR)

torch.manual_seed(1)

data = np.load(BASE_DIR + "\datasets\education\linear_train.npy")

# 데이터
x_train = torch.FloatTensor([[data[0]] for data in data])
y_train = torch.FloatTensor([[data[1]] for data in data])

# 모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000

# nn.Module을 사용하지 않은 것
def linear_without_nn():
    # optimizer 설정
    optimizer = optim.SGD([W, b], lr=0.0025)
    for epoch in range(nb_epochs + 1):

        # H(x) 계산
        hypothesis = x_train * W + b

        # cost 계산
        cost = torch.mean((hypothesis - y_train) ** 2)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 100번마다 로그 출력
        if epoch % 100 == 0:
            print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
                epoch, nb_epochs, W.item(), b.item(), cost.item()
            ))

# nn.Module을 사용한 것
def linear_with_nn():

    # 모델을 선언 및 초기화. 단순 선형 회귀이므로 input_dim=1, output_dim=1.
    model = nn.Linear(1, 1)
    print(list(model.parameters()))
    # optimizer 설정
    optimizer = optim.SGD(model.parameters(), lr=0.007)
    for epoch in range(nb_epochs + 1):

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

        # cost로 H(x) 개선하는 부분
        # gradient를 0으로 초기화
        optimizer.zero_grad()
        # 비용 함수를 미분하여 gradient 계산
        cost.backward() # backward 연산
        # W와 b를 업데이트
        optimizer.step()

        if epoch % 100 == 0:
            print(list(model.parameters()))
            # 100번마다 로그 출력
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, cost.item()
            ))

    return model

# 모델 그리기
def draw(model):
    W = list(list(model.parameters())[0])
    b = list(list(model.parameters())[1])
    # print("ddddddddddddd")
    # print(W)
    # print(b)
    # print("ddddddddddddddddd")
    x = np.arange(1, 20)
    # print(x)
    y = W * x + b
    # print(y)
    plt.plot(x, y)
    plt.show()

# GPU 사용
def useGPU(model, model_path):
    # 어떤 하드웨어 자원을 사용할 지
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data를 target 하드웨어로 copy
    data = data.to(device)

    # model을 target 하드웨어로 적용
    model.to(device)

    # 저장된 model을 불러올 때 어떤 device로 불러올지 지정
    model.load_state_dict(torch.load(model_path, map_location=device))


# model = TestModel()
# ''' training '''
#
# # save
# savePath = "./output/test_model.pth"
# torch.save(model.state_dict(), savePath)
#
# # load
# new_model = TestModel()
# new_model.load_state_dict(torch.load("./output/test_model.pth"))


if __name__ == "__main__":
    # linear_without_nn()
    draw(linear_with_nn())