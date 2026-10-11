"""Train a small CIFAR-10 CNN from scratch; run with: python train.py"""
import json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(min(8, torch.get_num_threads()))
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA=Path('data'); DATA.mkdir(exist_ok=True)
class CIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b): return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True))
        self.features=nn.Sequential(block(3,32),block(32,32),nn.MaxPool2d(2),nn.Dropout2d(.1),block(32,64),block(64,64),nn.MaxPool2d(2),nn.Dropout2d(.15),block(64,128),block(128,128),nn.MaxPool2d(2),nn.Dropout2d(.2))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(128*4*4,256),nn.ReLU(inplace=True),nn.Dropout(.35),nn.Linear(256,10))
    def forward(self,x): return self.classifier(self.features(x))
mean=(.4914,.4822,.4465); std=(.247,.243,.261)
train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean,std)])
test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
train_ds=datasets.CIFAR10(DATA,train=True,download=True,transform=train_tf); test_ds=datasets.CIFAR10(DATA,train=False,download=True,transform=test_tf)
train_dl=DataLoader(train_ds,batch_size=128,shuffle=True,num_workers=2); test_dl=DataLoader(test_ds,batch_size=256,shuffle=False,num_workers=2)
model=CIFARNet().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=12); criterion=nn.CrossEntropyLoss(); classes=train_ds.classes
start=time.time(); best=0.
for epoch in range(12):
    model.train(); loss_sum=correct=n=0
    for x,y in train_dl:
        x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(set_to_none=True); logits=model(x); loss=criterion(logits,y); loss.backward(); opt.step()
        loss_sum+=loss.item()*len(y); correct+=(logits.argmax(1)==y).sum().item(); n+=len(y)
    sched.step(); model.eval(); tc=tn=0
    with torch.no_grad():
        for x,y in test_dl:
            logits=model(x.to(DEVICE)); tc+=(logits.argmax(1).cpu()==y).sum().item(); tn+=len(y)
    acc=tc/tn; best=max(best,acc)
    print(f'Epoch {epoch+1:02d}/12 train_loss={loss_sum/n:.4f} train_acc={correct/n:.4f} test_acc={acc:.4f} elapsed={time.time()-start:.0f}s',flush=True)
    torch.save({'state_dict':model.state_dict(),'classes':classes,'mean':mean,'std':std,'epoch':epoch+1,'test_accuracy':acc},'cifar10_model.pt')
model.eval(); picks=[]
with torch.no_grad():
    for i in range(12):
        x,y=test_ds[i]; logits=model(x.unsqueeze(0).to(DEVICE))[0]; probs=logits.softmax(0); pred=int(probs.argmax())
        picks.append({'test_index':i,'actual':classes[y],'predicted':classes[pred],'confidence':round(float(probs[pred]),4),'correct':pred==y})
report={'dataset':'CIFAR-10','seed':SEED,'epochs':12,'device':str(DEVICE),'heldout_test_images':len(test_ds),'test_accuracy':tc/tn,'test_correct':tc,'best_epoch_test_accuracy':best,'sample_predictions':picks}
Path('results.json').write_text(json.dumps(report,indent=2)+'\n'); print('FINAL',json.dumps(report,indent=2))
