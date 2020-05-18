# test 1

```python
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
```

```text
torch.Size([60000, 28, 28])
torch.Size([10000, 28, 28])
cuda:0
Epoch: 001 Cost = 0.406162679
Epoch: 002 Cost = 0.260867000
Epoch: 003 Cost = 0.209885240
Epoch: 004 Cost = 0.168573573
Epoch: 005 Cost = 0.133219808
Epoch: 006 Cost = 0.106429964
Epoch: 007 Cost = 0.084967032
Epoch: 008 Cost = 0.069988668
Epoch: 009 Cost = 0.059502568
Epoch: 010 Cost = 0.048015110
Epoch: 011 Cost = 0.046649337
Epoch: 012 Cost = 0.038152520
Epoch: 013 Cost = 0.038395829
Epoch: 014 Cost = 0.034589957
Epoch: 015 Cost = 0.029597614
Epoch: 016 Cost = 0.028392624
Epoch: 017 Cost = 0.026605817
Epoch: 018 Cost = 0.023528371
Epoch: 019 Cost = 0.023793820
Epoch: 020 Cost = 0.020256151
소요 시간 : 733.30381 sec
12.22 minutes
Accuracy for 10000 images: 90.89%
```

# test 2

```python
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
            # nn.BatchNorm1d(120),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Linear(120, 60),
            # nn.BatchNorm1d(60),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Linear(60, 10)
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


NUM_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
```

```text
torch.Size([60000, 28, 28])
torch.Size([10000, 28, 28])
cuda:0
Epoch: 001 Cost = 0.651092827
Epoch: 002 Cost = 0.366382003
Epoch: 003 Cost = 0.311752290
Epoch: 004 Cost = 0.279024631
Epoch: 005 Cost = 0.254602790
Epoch: 006 Cost = 0.233815730
Epoch: 007 Cost = 0.216720030
Epoch: 008 Cost = 0.200766906
Epoch: 009 Cost = 0.186015129
Epoch: 010 Cost = 0.173966855
Epoch: 011 Cost = 0.161754891
Epoch: 012 Cost = 0.150055960
Epoch: 013 Cost = 0.139166728
Epoch: 014 Cost = 0.128652796
Epoch: 015 Cost = 0.119885758
Epoch: 016 Cost = 0.109571487
Epoch: 017 Cost = 0.099043339
Epoch: 018 Cost = 0.092232116
Epoch: 019 Cost = 0.082219340
Epoch: 020 Cost = 0.075753875
Epoch: 021 Cost = 0.068421543
Epoch: 022 Cost = 0.061678607
Epoch: 023 Cost = 0.057435043
Epoch: 024 Cost = 0.049315766
Epoch: 025 Cost = 0.043866612
Epoch: 026 Cost = 0.039933093
Epoch: 027 Cost = 0.034989342
Epoch: 028 Cost = 0.032460131
Epoch: 029 Cost = 0.027149357
Epoch: 030 Cost = 0.024550604
Epoch: 031 Cost = 0.019872699
Epoch: 032 Cost = 0.017848821
Epoch: 033 Cost = 0.016218157
Epoch: 034 Cost = 0.014691328
Epoch: 035 Cost = 0.011920276
Epoch: 036 Cost = 0.011418328
Epoch: 037 Cost = 0.010871250
Epoch: 038 Cost = 0.010676297
Epoch: 039 Cost = 0.007444708
Epoch: 040 Cost = 0.006359847
Epoch: 041 Cost = 0.009251953
Epoch: 042 Cost = 0.006416250
Epoch: 043 Cost = 0.004367306
Epoch: 044 Cost = 0.006702112
Epoch: 045 Cost = 0.004849672
Epoch: 046 Cost = 0.010617674
Epoch: 047 Cost = 0.004870383
Epoch: 048 Cost = 0.002920232
Epoch: 049 Cost = 0.001682734
Epoch: 050 Cost = 0.008068107
Epoch: 051 Cost = 0.007799657
Epoch: 052 Cost = 0.001899698
Epoch: 053 Cost = 0.001274555
Epoch: 054 Cost = 0.000928982
Epoch: 055 Cost = 0.005559484
Epoch: 056 Cost = 0.017763158
Epoch: 057 Cost = 0.002378537
Epoch: 058 Cost = 0.002297281
Epoch: 059 Cost = 0.004573561
Epoch: 060 Cost = 0.002040372
Epoch: 061 Cost = 0.001139711
Epoch: 062 Cost = 0.000886246
Epoch: 063 Cost = 0.001165634
Epoch: 064 Cost = 0.016704295
Epoch: 065 Cost = 0.002141198
Epoch: 066 Cost = 0.001164955
Epoch: 067 Cost = 0.000949084
Epoch: 068 Cost = 0.000529266
Epoch: 069 Cost = 0.000411811
Epoch: 070 Cost = 0.003097400
Epoch: 071 Cost = 0.021212768
Epoch: 072 Cost = 0.001564036
Epoch: 073 Cost = 0.000897475
Epoch: 074 Cost = 0.000494266
Epoch: 075 Cost = 0.000409386
Epoch: 076 Cost = 0.000364195
Epoch: 077 Cost = 0.000320974
Epoch: 078 Cost = 0.000272383
Epoch: 079 Cost = 0.022904141
Epoch: 080 Cost = 0.003503848
Epoch: 081 Cost = 0.001310397
Epoch: 082 Cost = 0.000960535
Epoch: 083 Cost = 0.001336146
Epoch: 084 Cost = 0.007680050
Epoch: 085 Cost = 0.002722531
Epoch: 086 Cost = 0.000656746
Epoch: 087 Cost = 0.000399281
Epoch: 088 Cost = 0.000355266
Epoch: 089 Cost = 0.000214741
Epoch: 090 Cost = 0.000189176
Epoch: 091 Cost = 0.000737405
Epoch: 092 Cost = 0.019228514
Epoch: 093 Cost = 0.001705151
Epoch: 094 Cost = 0.000661195
Epoch: 095 Cost = 0.000639444
Epoch: 096 Cost = 0.000324400
Epoch: 097 Cost = 0.000221955
Epoch: 098 Cost = 0.000211036
Epoch: 099 Cost = 0.007914647
Epoch: 100 Cost = 0.013803324
소요 시간 : 1647.20052 sec
27.45 minutes
Accuracy for 10000 images: 89.00%
```