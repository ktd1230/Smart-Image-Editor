import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

model = nn.Linear(1, 1)
model.load_state_dict(torch.load("../../checkpoints/model.pth"))
# eval()는 test mode로 model을 만들어줌..
model.eval()
xdata = np.load("../../../datasets/education/linear_test_x.npy")

xdata = np.expand_dims(xdata, axis=1)
test_data = torch.FloatTensor(xdata)
pred_y = model(test_data)
xdata = xdata.reshape(-1,)
pred_y = pred_y.reshape(-1,)
plt.plot(xdata.tolist(),pred_y.tolist())

plt.show()


