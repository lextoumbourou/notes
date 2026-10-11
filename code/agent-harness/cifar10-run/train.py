"""Train a small CIFAR-10 CNN from scratch and evaluate on the official test set.
Run: .venv/bin/python train.py [--epochs 5]
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout2d(.15),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout2d(.2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout2d(.25),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256*4*4, 256), nn.ReLU(inplace=True), nn.Dropout(.4), nn.Linear(256, 10))
    def forward(self, x): return self.classifier(self.features(x))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=128); a=p.parse_args()
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    torch.set_num_threads(min(8,torch.get_num_threads()))
    mean=(.4914,.4822,.4465); std=(.247,.243,.261)
    train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean,std)])
    test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
    train=datasets.CIFAR10('./data',train=True,download=True,transform=train_tf)
    test=datasets.CIFAR10('./data',train=False,download=True,transform=test_tf)
    tr=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=2,pin_memory=False)
    te=DataLoader(test,batch_size=256,shuffle=False,num_workers=2)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',dev,flush=True)
    model=SmallCNN().to(dev); opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=5e-4,nesterov=True)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs); loss_fn=nn.CrossEntropyLoss(label_smoothing=.05)
    for epoch in range(a.epochs):
        model.train(); seen=correct=0; total_loss=0.
        for x,y in tr:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(set_to_none=True); z=model(x); loss=loss_fn(z,y); loss.backward(); opt.step()
            total_loss+=loss.item()*y.size(0); correct+=(z.argmax(1)==y).sum().item(); seen+=y.size(0)
        sched.step(); print(f'Epoch {epoch+1}/{a.epochs}: train loss={total_loss/seen:.4f}, train acc={correct/seen:.4%}',flush=True)
    model.eval(); correct=total=0; predictions=[]
    with torch.no_grad():
        for x,y in te:
            z=model(x.to(dev)); prob=z.softmax(1).cpu(); pred=prob.argmax(1)
            correct+=(pred==y).sum().item(); total+=y.size(0)
            if len(predictions)<12:
                for i in range(min(12-len(predictions),len(y))):
                    predictions.append({'index':len(predictions),'true':CLASSES[y[i].item()],'predicted':CLASSES[pred[i].item()],'confidence':round(prob[i,pred[i]].item(),4)})
    accuracy=correct/total
    torch.save({'model_state_dict':model.cpu().state_dict(),'classes':CLASSES,'mean':mean,'std':std,'test_accuracy':accuracy},'cifar10_model.pt')
    report={'dataset':'CIFAR-10 official split','train_images':len(train),'test_images':total,'epochs':a.epochs,'test_correct':correct,'test_accuracy':accuracy,'sample_test_predictions':predictions}
    Path('results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'\nHeld-out official test accuracy: {correct}/{total} = {accuracy:.2%}')
    print('Sample test predictions (images are held-out):')
    for q in predictions: print(f"  #{q['index']:02d}: predicted {q['predicted']} ({q['confidence']:.1%}), actual {q['true']}")
if __name__=='__main__': main()
