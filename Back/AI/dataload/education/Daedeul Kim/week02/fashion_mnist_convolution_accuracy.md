# Fashion MNIST classification test result

**fashion_mnist_convolution.py** 파일을 통해서 테스트가 수행됨.

## Conv2d - 2개

### test 1

```python
# shape (?, 28, 28, 1)
# conv  (?, 28, 28, 16)
# pool  (?, 14, 14, 16)
self.layer1 = nn.Sequential(
	nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
	nn.BatchNorm2d(16),
	nn.ReLU(),
	nn.MaxPool2d(kernel_size=2)
)

# shape (?, 14, 14, 16)
# conv  (?, 14, 14, 32)
# pool  (?, 7, 7, 32)
self.layer2 = nn.Sequential(
	nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
	nn.BatchNorm2d(32),
	nn.ReLU(),
	nn.MaxPool2d(kernel_size=2)
)

self.layer3 = nn.Sequential(
	nn.Linear(7 * 7 * 32, 256),
	nn.BatchNorm1d(256),
	nn.ReLU()
)
self.layer4 = nn.Sequential(
	nn.Linear(256, 64),
	nn.BatchNorm1d(64),
	nn.ReLU()
)
self.layer5 = nn.Sequential(
	nn.Linear(64, 10)
)

def forward(self, x):
	out = self.layer1(x)
	out = self.layer2(out)
	out = out.view(out.size(0), -1)
	out = self.layer3(out)
	out = self.layer4(out)
	out = self.layer5(out)
	return out
```

```python
NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001
```

```text
torch.Size([60000, 28, 28])
torch.Size([10000, 28, 28])
cuda:0
Epoch: 001 Cost = 0.468485892
Epoch: 002 Cost = 0.299410313
Epoch: 003 Cost = 0.251036733
Epoch: 004 Cost = 0.218887091
Epoch: 005 Cost = 0.195228577
Epoch: 006 Cost = 0.176532775
Epoch: 007 Cost = 0.156352550
Epoch: 008 Cost = 0.146707267
Epoch: 009 Cost = 0.129799291
Epoch: 010 Cost = 0.118753076
Epoch: 011 Cost = 0.109231927
Epoch: 012 Cost = 0.098587625
Epoch: 013 Cost = 0.090484083
Epoch: 014 Cost = 0.079831176
Epoch: 015 Cost = 0.073483586
Epoch: 016 Cost = 0.067210525
Epoch: 017 Cost = 0.060664881
Epoch: 018 Cost = 0.054550912
Epoch: 019 Cost = 0.051188994
Epoch: 020 Cost = 0.045649290
소요 시간 : 758.97524 sec
12.65 minutes
Accuracy for 10000 images: 91.52%
```

## Conv2d - 3개

## test 1

```python
# shape (?, 28, 28, 1)
# conv  (?, 28, 28, 16)
# pool  (?, 14, 14, 16)
self.layer1 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2)
)

# shape (?, 14, 14, 16)
# conv  (?, 14, 14, 32)
# pool  (?, 7, 7, 32)
self.layer2 = nn.Sequential(
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2)
)

# shape (?, 7, 7, 32)
# conv  (?, 7, 7, 64)
# pool  (?, 3, 3, 64)
self.conv3 = nn.Sequential(
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2)
)

self.layer3 = nn.Sequential(
    nn.Linear(3 * 3 * 64, 256),
    nn.BatchNorm1d(256),
    nn.ReLU()
)
self.layer4 = nn.Sequential(
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU()
)
self.layer5 = nn.Sequential(
    nn.Linear(128, 10),
)
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
```

```text
torch.Size([60000, 28, 28])
torch.Size([10000, 28, 28])
cuda:0
Epoch: 001 Cost = 1.291537523
Epoch: 002 Cost = 0.483429492
Epoch: 003 Cost = 0.341359705
Epoch: 004 Cost = 0.278577030
Epoch: 005 Cost = 0.237164706
Epoch: 006 Cost = 0.206295371
Epoch: 007 Cost = 0.180732101
Epoch: 008 Cost = 0.160681874
Epoch: 009 Cost = 0.142848477
Epoch: 010 Cost = 0.124191277
Epoch: 011 Cost = 0.114321023
Epoch: 012 Cost = 0.100906506
Epoch: 013 Cost = 0.089115202
Epoch: 014 Cost = 0.085168526
Epoch: 015 Cost = 0.076282188
Epoch: 016 Cost = 0.067545176
Epoch: 017 Cost = 0.065273859
Epoch: 018 Cost = 0.056527950
Epoch: 019 Cost = 0.054204833
Epoch: 020 Cost = 0.050378811
소요 시간 : 677.64402 sec
11.29 minutes
Accuracy for 10000 images: 91.80%
```