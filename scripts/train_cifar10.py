#!/usr/bin/env python3
"""Optional real ImageNet->CIFAR-10 experiment; downloads data/weights when run.
Not part of the offline benchmark; never use offline metrics to describe this run.
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
from sklearn.model_selection import train_test_split
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from visionforge.models import CustomCNN,make_resnet18
from visionforge.engine import fit,predict
from visionforge.metrics import classification_metrics

class FreezeBackbone(torch.nn.Module):
    """Keep a frozen ImageNet trunk in inference mode while training classification head."""
    def __init__(self,model):
        super().__init__();self.model=model
        for p in self.model.parameters(): p.requires_grad_(False)
        for p in self.model.fc.parameters(): p.requires_grad_(True)
    def train(self,mode=True):
        super().train(mode)
        self.model.eval();self.model.fc.train(mode)
        return self
    def forward(self,x): return self.model(x)

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--data-dir',default='data');pa.add_argument('--out',default='experiments/cifar10_imagenet')
    pa.add_argument('--seed',type=int,default=42);pa.add_argument('--train-limit',type=int,default=12000)
    pa.add_argument('--cnn-epochs',type=int,default=10);pa.add_argument('--transfer-epochs',type=int,default=5)
    pa.add_argument('--batch-size',type=int,default=64);pa.add_argument('--device',default='auto')
    a=pa.parse_args();torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    device='cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device)
    # Identical ImageNet normalization for the two competing architectures.
    tr=transforms.Compose([transforms.Resize((96,96)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    ev=transforms.Compose([transforms.Resize((96,96)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    ds_aug=datasets.CIFAR10(root=a.data_dir,train=True,download=True,transform=tr)
    ds_clean=datasets.CIFAR10(root=a.data_dir,train=True,download=False,transform=ev)
    ds_test=datasets.CIFAR10(root=a.data_dir,train=False,download=True,transform=ev)
    ix=np.arange(len(ds_aug.targets))
    pool,valid=train_test_split(ix,test_size=.15,random_state=a.seed,stratify=ds_aug.targets)
    if a.train_limit and a.train_limit<len(pool):
        pool,_=train_test_split(pool,train_size=a.train_limit,random_state=a.seed,stratify=np.array(ds_aug.targets)[pool])
    train_loader=DataLoader(Subset(ds_aug,pool),batch_size=a.batch_size,shuffle=True,num_workers=2)
    val_loader=DataLoader(Subset(ds_clean,valid),batch_size=a.batch_size,num_workers=2)
    test_loader=DataLoader(ds_test,batch_size=a.batch_size,num_workers=2)
    out=Path(a.out);(out/'artifacts').mkdir(exist_ok=True,parents=True)
    meta={'status':'complete','experiment':'CIFAR-10; ImageNet-1K ResNet18 head-only transfer','seed':a.seed,
          'train_examples':len(pool),'validation_examples':len(valid),'test_examples':len(ds_test),
          'pretraining':'TorchVision ResNet18_Weights.IMAGENET1K_V1; weights download required; frozen backbone',
          'image_size':96,'models':{},'device':device,'validation_policy':'selected by validation accuracy; test measured once per model'}
    for model_id in ['custom_cnn','resnet18_imagenet']:
        if model_id=='custom_cnn':
            model=CustomCNN(3,10);epochs=a.cnn_epochs;lr=.001
        else:
            model=FreezeBackbone(make_resnet18(10,mode='imagenet',pretrained=True));epochs=a.transfer_epochs;lr=.003
        history,best=fit(model,train_loader,val_loader,epochs=epochs,lr=lr,device=device,tag=model_id)
        y,p=predict(model,test_loader,device)
        meta['models'][model_id]={'history':history,'best_epoch':best,'metrics':classification_metrics(y,p,a.seed)}
        state=model.model.state_dict() if isinstance(model,FreezeBackbone) else model.state_dict()
        torch.save({'model':state,'architecture':model_id,'input_size':96,'normalization':'ImageNet'},out/'artifacts'/f'{model_id}.pt')
        np.savez_compressed(out/'artifacts'/f'{model_id}_predictions.npz',y_true=y,probabilities=p)
    (out/'results.json').write_text(json.dumps(meta,indent=2))
    print('Verified CIFAR10 study saved in',out)

if __name__=='__main__':main()
