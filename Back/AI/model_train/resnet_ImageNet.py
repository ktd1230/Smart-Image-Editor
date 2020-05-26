from models.resnet import ResNet
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets,transforms
from transformers import AdamW,get_linear_schedule_with_warmup
from tqdm import trange,tqdm
import json
import pickle

class ResnetCls(nn.Module):

    def __init__(self,resnet_model,num_class):
        super(ResnetCls,self).__init__()
        self.feature = resnet_model
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(2048,num_class)

    def forward(self,x):
        out = self.feature(x)
        out = self.pool(out)
        out = out.view(out.shape[0],-1)
        out = self.linear(out)
        return out

def updateBn(model,decay_param):
    for m in model.parameters():
        if isinstance(m,nn.BatchNorm2d):
            m.weight.grad.data.add_(decay_param*torch.sign(m.weight.data))

def train(model,epochs,batch):
    train_loader = torch.utils.data.DataLoader(
        datasets.imagenet(root='datasets/',train=True, download=True,
                          transform=transforms.Compose([
                              transforms.RandomResizedCrop(224),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                          ])),
        batch_size=batch, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=0.1, eps=1e-8)
    num_training_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    loss_function = nn.CrossEntropyLoss()
    global_step = 0
    epochs_trained = 0

    tr_loss = 0.0
    logging_loss = 0.0
    train_iterator = trange(
        epochs_trained, int(epochs), desc="Epoch"
    )
    logging_steps = 500
    loss_record = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        model.train()
        for idx_of_batch,(data, target) in enumerate(epoch_iterator):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            loss.backward()
            updateBn(model,0.00001)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
            global_step += 1
            if logging_steps > 0 and global_step % logging_steps == 0:
                logs = {}
                loss_scalar = (tr_loss - logging_loss) / logging_steps
                learning_rate_scalar = scheduler.get_last_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                loss_record.append(loss_scalar)
                logging_loss = tr_loss
                epoch_iterator.write(json.dumps({**logs, **{"step": global_step}}))
    return loss_record,model

def test(model,batch):
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='datasets/',train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])),
        batch_size=batch, shuffle=True)

if __name__ == "__main__":
    backbone = ResNet([3,4,23,3])
    model = ResnetCls(backbone,1000)
    loss,trained_model = train(model,1,52)
    with open('loss.pkl','wb') as f:
        pickle.dump(loss,f)
    torch.save(model.state_dict(),"resnet_101.pth")

