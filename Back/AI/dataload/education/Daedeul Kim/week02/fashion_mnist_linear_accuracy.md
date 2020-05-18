# Fashion MNIST classification test result

**fashion_mnist_linear.py** 파일을 통해서 테스트가 수행됨.

```text
batch_size = 100
num_epochs = 20
learning_rate = 0.0001
```

## Layer 3 

### test 1 (No ReLU)

```python
self.layer1 = nn.Sequential(
    torch.nn.Linear(784, 256, bias=True),
    torch.nn.BatchNorm1d(256),
    # torch.nn.ReLU()
)

self.layer2 = nn.Sequential(
    torch.nn.Linear(256, 64, bias=True),
    torch.nn.BatchNorm1d(64),
    # torch.nn.ReLU()
)

self.layer3 = nn.Sequential(
    torch.nn.Linear(64, 8, bias=True)
)
```

```text
torch.Size([60000, 28, 28])
torch.Size([10000, 28, 28])
Epoch: 001 Cost = 1.207170844
Epoch: 002 Cost = 1.056105614
Epoch: 003 Cost = 1.002200484
Epoch: 004 Cost = 0.969641566
Epoch: 005 Cost = 0.946426034
Epoch: 006 Cost = 0.928422391
Epoch: 007 Cost = 0.918688893
Epoch: 008 Cost = 0.912511945
Epoch: 009 Cost = 0.907609761
Epoch: 010 Cost = 0.899722934
Epoch: 011 Cost = 0.901443303
Epoch: 012 Cost = 0.896398246
Epoch: 013 Cost = 0.896608591
Epoch: 014 Cost = 0.892547309
Epoch: 015 Cost = 0.889116764
Epoch: 016 Cost = 0.888523698
Epoch: 017 Cost = 0.886315823
Epoch: 018 Cost = 0.887891173
Epoch: 019 Cost = 0.882654846
Epoch: 020 Cost = 0.884824574
소요 시간 : 236.52089sec
Accuracy for 10000 images: 65.03%
```

### test 2 (with ReLU)

```python
self.layer1 = nn.Sequential(
    torch.nn.Linear(784, 256, bias=True),
    torch.nn.BatchNorm1d(256),
    torch.nn.ReLU()
)

self.layer2 = nn.Sequential(
    torch.nn.Linear(256, 64, bias=True),
    torch.nn.BatchNorm1d(64),
    torch.nn.ReLU()
)

self.layer3 = nn.Sequential(
    torch.nn.Linear(64, 8, bias=True)
)
```

```text
torch.Size([60000, 28, 28])
torch.Size([10000, 28, 28])
Epoch: 001 Cost = 1.113665581
Epoch: 002 Cost = 0.893351376
Epoch: 003 Cost = 0.841124952
Epoch: 004 Cost = 0.811440289
Epoch: 005 Cost = 0.792192161
Epoch: 006 Cost = 0.779539943
Epoch: 007 Cost = 0.769473493
Epoch: 008 Cost = 0.760152578
Epoch: 009 Cost = 0.745657444
Epoch: 010 Cost = 0.739479959
Epoch: 011 Cost = 0.735363007
Epoch: 012 Cost = 0.725894094
Epoch: 013 Cost = 0.718340576
Epoch: 014 Cost = 0.712965786
Epoch: 015 Cost = 0.710429847
Epoch: 016 Cost = 0.705032229
Epoch: 017 Cost = 0.695755363
Epoch: 018 Cost = 0.692446172
Epoch: 019 Cost = 0.686132848
Epoch: 020 Cost = 0.680829048
소요 시간 : 243.01313sec
Accuracy for 10000 images: 69.35%
```

## Layer 4 (with ReLU)

### test 1

```python
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
    torch.nn.Linear(64, 8, bias=True)
)
```

```text
torch.Size([60000, 28, 28])
torch.Size([10000, 28, 28])
Epoch: 001 Cost = 1.104004264
Epoch: 002 Cost = 0.879830360
Epoch: 003 Cost = 0.825729191
Epoch: 004 Cost = 0.794785440
Epoch: 005 Cost = 0.779814124
Epoch: 006 Cost = 0.763151169
Epoch: 007 Cost = 0.746529758
Epoch: 008 Cost = 0.738618553
Epoch: 009 Cost = 0.728139877
Epoch: 010 Cost = 0.717194498
Epoch: 011 Cost = 0.714296937
Epoch: 012 Cost = 0.702905297
Epoch: 013 Cost = 0.693738222
Epoch: 014 Cost = 0.685729504
Epoch: 015 Cost = 0.678414702
Epoch: 016 Cost = 0.671905518
Epoch: 017 Cost = 0.664673567
Epoch: 018 Cost = 0.661363780
Epoch: 019 Cost = 0.653964579
Epoch: 020 Cost = 0.649090290
소요 시간 : 261.25sec
Accuracy for 10000 images: 69.87%
```


## Layer 5 (with ReLU)

### test 1

```python
self.layer1 = nn.Sequential(
            torch.nn.Linear(784, 512, bias=True),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU()
)

self.layer2 = nn.Sequential(
    torch.nn.Linear(512, 256, bias=True),
    torch.nn.BatchNorm1d(256),
    torch.nn.ReLU()
)

self.layer3 = nn.Sequential(
    torch.nn.Linear(256, 128, bias=True),
    torch.nn.BatchNorm1d(128),
    torch.nn.ReLU()
)

self.layer4 = nn.Sequential(
    torch.nn.Linear(128, 64, bias=True),
    torch.nn.BatchNorm1d(64),
    torch.nn.ReLU()
)

self.layer5 = nn.Sequential(
    torch.nn.Linear(64, 32, bias=True)
)
```

```text
torch.Size([60000, 28, 28])
torch.Size([10000, 28, 28])
Epoch: 001 Cost = 1.057777882
Epoch: 002 Cost = 0.850647509
Epoch: 003 Cost = 0.801358521
Epoch: 004 Cost = 0.766346276
Epoch: 005 Cost = 0.744191408
Epoch: 006 Cost = 0.729273677
Epoch: 007 Cost = 0.715010166
Epoch: 008 Cost = 0.698178172
Epoch: 009 Cost = 0.685983181
Epoch: 010 Cost = 0.674582005
Epoch: 011 Cost = 0.663600862
Epoch: 012 Cost = 0.652157128
Epoch: 013 Cost = 0.646354496
Epoch: 014 Cost = 0.635855734
Epoch: 015 Cost = 0.630743086
Epoch: 016 Cost = 0.623735964
Epoch: 017 Cost = 0.617698491
Epoch: 018 Cost = 0.610510826
Epoch: 019 Cost = 0.604346812
Epoch: 020 Cost = 0.600911677
Accuracy for 10000 images: 69.92%
```

### test 2

```python
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
    torch.nn.Linear(32, 16, bias=True)
)
```

```text
torch.Size([60000, 28, 28])
torch.Size([10000, 28, 28])
Epoch: 001 Cost = 1.210536242
Epoch: 002 Cost = 0.613714993
Epoch: 003 Cost = 0.458107501
Epoch: 004 Cost = 0.382254481
Epoch: 005 Cost = 0.338944435
Epoch: 006 Cost = 0.306561142
Epoch: 007 Cost = 0.283756047
Epoch: 008 Cost = 0.260798126
Epoch: 009 Cost = 0.245537594
Epoch: 010 Cost = 0.229344368
Epoch: 011 Cost = 0.213971823
Epoch: 012 Cost = 0.200146154
Epoch: 013 Cost = 0.192467943
Epoch: 014 Cost = 0.178946212
Epoch: 015 Cost = 0.169641137
Epoch: 016 Cost = 0.161499396
Epoch: 017 Cost = 0.151354074
Epoch: 018 Cost = 0.142755911
Epoch: 019 Cost = 0.137448609
Epoch: 020 Cost = 0.130794510
Accuracy for 10000 images: 88.39%
```

```text
torch.Size([60000, 28, 28])
torch.Size([10000, 28, 28])
Epoch: 001 Cost = 1.201605082
Epoch: 002 Cost = 0.608586729
Epoch: 003 Cost = 0.453705877
Epoch: 004 Cost = 0.381338686
Epoch: 005 Cost = 0.334718287
Epoch: 006 Cost = 0.302471101
Epoch: 007 Cost = 0.277955055
Epoch: 008 Cost = 0.259107530
Epoch: 009 Cost = 0.240986660
Epoch: 010 Cost = 0.227537498
Epoch: 011 Cost = 0.214474037
Epoch: 012 Cost = 0.199368790
Epoch: 013 Cost = 0.189676359
Epoch: 014 Cost = 0.177696154
Epoch: 015 Cost = 0.168187454
Epoch: 016 Cost = 0.160694793
Epoch: 017 Cost = 0.152387425
Epoch: 018 Cost = 0.141590863
Epoch: 019 Cost = 0.136434257
Epoch: 020 Cost = 0.127704382
소요 시간 : 293.34293sec
Accuracy for 10000 images: 88.86%
```

### test 3

```python
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
    torch.nn.Linear(32, 8, bias=True)  # 16->8
)
```

```text
torch.Size([60000, 28, 28])
torch.Size([10000, 28, 28])
Epoch: 001 Cost = 1.213869333
Epoch: 002 Cost = 0.942492068
Epoch: 003 Cost = 0.862751603
Epoch: 004 Cost = 0.823272705
Epoch: 005 Cost = 0.794175506
Epoch: 006 Cost = 0.776331067
Epoch: 007 Cost = 0.759932458
Epoch: 008 Cost = 0.743654132
Epoch: 009 Cost = 0.733438134
Epoch: 010 Cost = 0.723478556
Epoch: 011 Cost = 0.717576206
Epoch: 012 Cost = 0.705515981
Epoch: 013 Cost = 0.696950972
Epoch: 014 Cost = 0.687002361
Epoch: 015 Cost = 0.684174895
Epoch: 016 Cost = 0.675122261
Epoch: 017 Cost = 0.668204486
Epoch: 018 Cost = 0.664678335
Epoch: 019 Cost = 0.660783231
Epoch: 020 Cost = 0.655986130
소요 시간 : 286.21918sec
Accuracy for 10000 images: 69.97%
```

### test 4
`num_epochs = 200`

```text
layer5 설정은 test 2와 동일함
```

```text
torch.Size([60000, 28, 28])
torch.Size([10000, 28, 28])
Epoch: 001 Cost = 1.162891626
Epoch: 002 Cost = 0.587491512
Epoch: 003 Cost = 0.441364616
Epoch: 004 Cost = 0.372894138
Epoch: 005 Cost = 0.330705762
Epoch: 006 Cost = 0.300927788
Epoch: 007 Cost = 0.277767062
Epoch: 008 Cost = 0.258603156
Epoch: 009 Cost = 0.241234973
Epoch: 010 Cost = 0.225036189
Epoch: 011 Cost = 0.211955160
Epoch: 012 Cost = 0.201832309
Epoch: 013 Cost = 0.188909352
Epoch: 014 Cost = 0.177973330
Epoch: 015 Cost = 0.169655994
Epoch: 016 Cost = 0.159908533
Epoch: 017 Cost = 0.152184382
Epoch: 018 Cost = 0.143950865
Epoch: 019 Cost = 0.138447165
Epoch: 020 Cost = 0.131147534
Epoch: 021 Cost = 0.124878302
Epoch: 022 Cost = 0.120209008
Epoch: 023 Cost = 0.111868151
Epoch: 024 Cost = 0.106853783
Epoch: 025 Cost = 0.103476211
Epoch: 026 Cost = 0.098541647
Epoch: 027 Cost = 0.092455089
Epoch: 028 Cost = 0.088417992
Epoch: 029 Cost = 0.083315700
Epoch: 030 Cost = 0.080288909
Epoch: 031 Cost = 0.077726141
Epoch: 032 Cost = 0.073606268
Epoch: 033 Cost = 0.070159212
Epoch: 034 Cost = 0.067883782
Epoch: 035 Cost = 0.065876074
Epoch: 036 Cost = 0.062889829
Epoch: 037 Cost = 0.058381107
Epoch: 038 Cost = 0.058927424
Epoch: 039 Cost = 0.056561407
Epoch: 040 Cost = 0.053612061
Epoch: 041 Cost = 0.050608069
Epoch: 042 Cost = 0.051935524
Epoch: 043 Cost = 0.047415946
Epoch: 044 Cost = 0.047162611
Epoch: 045 Cost = 0.044292405
Epoch: 046 Cost = 0.041150142
Epoch: 047 Cost = 0.045439716
Epoch: 048 Cost = 0.042083602
Epoch: 049 Cost = 0.041767139
Epoch: 050 Cost = 0.039237894
Epoch: 051 Cost = 0.037968196
Epoch: 052 Cost = 0.036556795
Epoch: 053 Cost = 0.035823662
Epoch: 054 Cost = 0.033774890
Epoch: 055 Cost = 0.032777384
Epoch: 056 Cost = 0.031980678
Epoch: 057 Cost = 0.032735307
Epoch: 058 Cost = 0.030941013
Epoch: 059 Cost = 0.031917516
Epoch: 060 Cost = 0.032543875
Epoch: 061 Cost = 0.031848513
Epoch: 062 Cost = 0.026398376
Epoch: 063 Cost = 0.029235434
Epoch: 064 Cost = 0.026489627
Epoch: 065 Cost = 0.027030943
Epoch: 066 Cost = 0.025352910
Epoch: 067 Cost = 0.026523046
Epoch: 068 Cost = 0.025716605
Epoch: 069 Cost = 0.025652923
Epoch: 070 Cost = 0.025445333
Epoch: 071 Cost = 0.024395848
Epoch: 072 Cost = 0.022026518
Epoch: 073 Cost = 0.025337402
Epoch: 074 Cost = 0.023545664
Epoch: 075 Cost = 0.020874914
Epoch: 076 Cost = 0.022859257
Epoch: 077 Cost = 0.022454433
Epoch: 078 Cost = 0.022145581
Epoch: 079 Cost = 0.021056399
Epoch: 080 Cost = 0.021436973
Epoch: 081 Cost = 0.019911004
Epoch: 082 Cost = 0.019971188
Epoch: 083 Cost = 0.022334140
Epoch: 084 Cost = 0.021647740
Epoch: 085 Cost = 0.018705193
Epoch: 086 Cost = 0.020523842
Epoch: 087 Cost = 0.019343140
Epoch: 088 Cost = 0.019198410
Epoch: 089 Cost = 0.019838385
Epoch: 090 Cost = 0.021350002
Epoch: 091 Cost = 0.019916929
Epoch: 092 Cost = 0.018357513
Epoch: 093 Cost = 0.017418545
Epoch: 094 Cost = 0.018002747
Epoch: 095 Cost = 0.015337640
Epoch: 096 Cost = 0.017793505
Epoch: 097 Cost = 0.015179237
Epoch: 098 Cost = 0.017980905
Epoch: 099 Cost = 0.019532057
Epoch: 100 Cost = 0.015748309
Epoch: 101 Cost = 0.016398942
Epoch: 102 Cost = 0.018386971
Epoch: 103 Cost = 0.017090427
Epoch: 104 Cost = 0.015827341
Epoch: 105 Cost = 0.015660750
Epoch: 106 Cost = 0.015314214
Epoch: 107 Cost = 0.014794806
Epoch: 108 Cost = 0.016986651
Epoch: 109 Cost = 0.015408264
Epoch: 110 Cost = 0.014149672
Epoch: 111 Cost = 0.017476570
Epoch: 112 Cost = 0.014649490
Epoch: 113 Cost = 0.015252324
Epoch: 114 Cost = 0.014617329
Epoch: 115 Cost = 0.014565146
Epoch: 116 Cost = 0.014126067
Epoch: 117 Cost = 0.013434304
Epoch: 118 Cost = 0.014863337
Epoch: 119 Cost = 0.016491698
Epoch: 120 Cost = 0.014271850
Epoch: 121 Cost = 0.013989855
Epoch: 122 Cost = 0.012008494
Epoch: 123 Cost = 0.016377036
Epoch: 124 Cost = 0.015422308
Epoch: 125 Cost = 0.013642664
Epoch: 126 Cost = 0.012407589
Epoch: 127 Cost = 0.015086336
Epoch: 128 Cost = 0.012645797
Epoch: 129 Cost = 0.013155483
Epoch: 130 Cost = 0.012660043
Epoch: 131 Cost = 0.012686653
Epoch: 132 Cost = 0.013192057
Epoch: 133 Cost = 0.013687614
Epoch: 134 Cost = 0.009746653
Epoch: 135 Cost = 0.015470540
Epoch: 136 Cost = 0.012979485
Epoch: 137 Cost = 0.012373016
Epoch: 138 Cost = 0.011804857
Epoch: 139 Cost = 0.013904128
Epoch: 140 Cost = 0.012157108
Epoch: 141 Cost = 0.013090886
Epoch: 142 Cost = 0.012260422
Epoch: 143 Cost = 0.012634345
Epoch: 144 Cost = 0.013008382
Epoch: 145 Cost = 0.012178616
Epoch: 146 Cost = 0.009836334
Epoch: 147 Cost = 0.011284672
Epoch: 148 Cost = 0.011243546
Epoch: 149 Cost = 0.011546589
Epoch: 150 Cost = 0.011249359
Epoch: 151 Cost = 0.011238500
Epoch: 152 Cost = 0.009573590
Epoch: 153 Cost = 0.013756932
Epoch: 154 Cost = 0.008261251
Epoch: 155 Cost = 0.012127231
Epoch: 156 Cost = 0.013712448
Epoch: 157 Cost = 0.010109068
Epoch: 158 Cost = 0.011891028
Epoch: 159 Cost = 0.009022149
Epoch: 160 Cost = 0.011148862
Epoch: 161 Cost = 0.009545069
Epoch: 162 Cost = 0.013641140
Epoch: 163 Cost = 0.010326000
Epoch: 164 Cost = 0.011518685
Epoch: 165 Cost = 0.009728874
Epoch: 166 Cost = 0.009337142
Epoch: 167 Cost = 0.009063600
Epoch: 168 Cost = 0.011510058
Epoch: 169 Cost = 0.009700834
Epoch: 170 Cost = 0.009520320
Epoch: 171 Cost = 0.010350059
Epoch: 172 Cost = 0.010861666
Epoch: 173 Cost = 0.010479167
Epoch: 174 Cost = 0.011347737
Epoch: 175 Cost = 0.007679192
Epoch: 176 Cost = 0.009809349
Epoch: 177 Cost = 0.009332608
Epoch: 178 Cost = 0.011794926
Epoch: 179 Cost = 0.009259737
Epoch: 180 Cost = 0.008550517
Epoch: 181 Cost = 0.009256613
Epoch: 182 Cost = 0.008799530
Epoch: 183 Cost = 0.009417672
Epoch: 184 Cost = 0.008084167
Epoch: 185 Cost = 0.011526578
Epoch: 186 Cost = 0.008401116
Epoch: 187 Cost = 0.009012152
Epoch: 188 Cost = 0.013418004
Epoch: 189 Cost = 0.009287437
Epoch: 190 Cost = 0.008120405
Epoch: 191 Cost = 0.009400396
Epoch: 192 Cost = 0.007530128
Epoch: 193 Cost = 0.012187423
Epoch: 194 Cost = 0.007819796
Epoch: 195 Cost = 0.008264175
Epoch: 196 Cost = 0.010126227
Epoch: 197 Cost = 0.009067599
Epoch: 198 Cost = 0.006833945
Epoch: 199 Cost = 0.009487892
Epoch: 200 Cost = 0.009389706
소요 시간 : 4884.75705sec
Accuracy for 10000 images: 88.64%
```

```text
# self.layer5 = nn.Sequential(
#     torch.nn.Linear(32, 10, bias=True)
# )
torch.Size([60000, 28, 28])
torch.Size([10000, 28, 28])
Epoch: 001 Cost = 0.882179677
Epoch: 002 Cost = 0.498057306
Epoch: 003 Cost = 0.403864622
Epoch: 004 Cost = 0.350658029
Epoch: 005 Cost = 0.316215903
Epoch: 006 Cost = 0.290797919
Epoch: 007 Cost = 0.268672645
Epoch: 008 Cost = 0.249113694
Epoch: 009 Cost = 0.237592548
Epoch: 010 Cost = 0.221183881
Epoch: 011 Cost = 0.206522688
Epoch: 012 Cost = 0.196364179
Epoch: 013 Cost = 0.186195031
Epoch: 014 Cost = 0.175648421
Epoch: 015 Cost = 0.164974704
Epoch: 016 Cost = 0.156944364
Epoch: 017 Cost = 0.147103012
Epoch: 018 Cost = 0.140516430
Epoch: 019 Cost = 0.133908078
Epoch: 020 Cost = 0.126349896
소요 시간 : 284.57736sec
Accuracy for 10000 images: 88.45%
```

## Layer 6 (with ReLU)

### test 1

```python
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
    torch.nn.Linear(32, 16, bias=True),
    torch.nn.BatchNorm1d(16),
    torch.nn.ReLU()
)

self.layer6 = nn.Sequential(
    torch.nn.Linear(16, 8, bias=True)
)
```

```text
torch.Size([60000, 28, 28])
torch.Size([10000, 28, 28])
Epoch: 001 Cost = 1.389370322
Epoch: 002 Cost = 1.066215634
Epoch: 003 Cost = 0.947344303
Epoch: 004 Cost = 0.874166369
Epoch: 005 Cost = 0.834712505
Epoch: 006 Cost = 0.801984191
Epoch: 007 Cost = 0.782338798
Epoch: 008 Cost = 0.765544832
Epoch: 009 Cost = 0.747116387
Epoch: 010 Cost = 0.734927893
Epoch: 011 Cost = 0.724318147
Epoch: 012 Cost = 0.711638153
Epoch: 013 Cost = 0.702207148
Epoch: 014 Cost = 0.695295751
Epoch: 015 Cost = 0.683164597
Epoch: 016 Cost = 0.676311076
Epoch: 017 Cost = 0.675903916
Epoch: 018 Cost = 0.663910925
Epoch: 019 Cost = 0.659340680
Epoch: 020 Cost = 0.652440965
소요 시간 : 320.25574sec
Accuracy for 10000 images: 69.65%
```