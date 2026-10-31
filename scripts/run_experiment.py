#!/usr/bin/env python3
"""Run the honest, offline-controlled digit benchmark; no internet required."""
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from visionforge.data import load_digit_splits
from visionforge.models import CustomCNN, make_resnet18
from visionforge.engine import fit, predict, RotationPretext
from visionforge.metrics import classification_metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cnn-epochs',type=int,default=15)
    p.add_argument('--pretext-epochs',type=int,default=2)
    p.add_argument('--transfer-epochs',type=int,default=6)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=3)
    p.add_argument('--out',default='experiments/offline_digits')
    args=p.parse_args()
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'artifacts').mkdir(exist_ok=True)
    loaders,x,y,splits=load_digit_splits(seed=args.seed,batch_size=args.batch_size)
    result={'experiment':'offline_digits','dataset':'scikit-learn digits (1797 8x8 grayscale images, upsampled to 32x32)',
           'protocol':'stratified 65/17.5/17.5 train/validation/test; single seed; test assessed after validation-based selection; fixed split for both models',
           'pretraining':'Four-way (0/90/180/270 degree) rotation prediction using TRAIN split only, not ImageNet; supervised fine-tuning on train; test never used for model selection',
           'environment':{'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,'accelerator':'cpu','threads':args.threads},
           'seed':args.seed,'split_sizes':{k:len(v) for k,v in splits.items()},'models':{},'configs':vars(args),'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    start=time.time()
    cnn=CustomCNN()
    hc,be=fit(cnn,loaders['train'],loaders['val'],epochs=args.cnn_epochs,lr=.002,tag='Custom CNN')
    yt,prob=predict(cnn,loaders['test'])
    result['models']['custom_cnn']={'display_name':'Custom CNN · trained from scratch','history':hc,'best_epoch':be,'metrics':classification_metrics(yt,prob,args.seed)}
    torch.save({'model':cnn.state_dict(),'architecture':'custom_cnn','num_classes':10,'input_size':32,'seed':args.seed},out/'artifacts'/'custom_cnn.pt')
    print('CUSTOM CNN held-out accuracy:',result['models']['custom_cnn']['metrics']['accuracy'],flush=True)
    np.savez_compressed(out/'artifacts'/'custom_cnn_predictions.npz',y_true=yt,probabilities=prob,test_indices=splits['test'])
    del cnn

    # Source task: use only training partition for self-supervised rotation classification.
    backbone=make_resnet18(num_classes=4,mode='digits')
    rot_ds=RotationPretext(loaders['train'].dataset)
    rot_loader=DataLoader(rot_ds,batch_size=args.batch_size,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(args.seed))
    hrot,brot=fit(backbone,rot_loader,loaders['val'] if False else rot_loader,epochs=args.pretext_epochs,lr=.001,tag='Rotation source')
    # Source loss/accuracy are TRAIN-partition scores, NOT external validation evidence.
    torch.save({'model':{k:v.half() if v.is_floating_point() else v for k,v in backbone.state_dict().items()},'architecture':'resnet18_digit_rotation_pretext','note':'rotation-source task weights; TRAIN partition only'},out/'artifacts'/'rotation_source_weights.pt')
    backbone.fc=torch.nn.Linear(backbone.fc.in_features,10)
    ht,bt=fit(backbone,loaders['train'],loaders['val'],epochs=args.transfer_epochs,lr=.0007,tag='Transfer ResNet18')
    yt2,prob2=predict(backbone,loaders['test'])
    result['models']['rotation_resnet18']={'display_name':'ResNet-18 · rotation-pretrained transfer','history':ht,'pretrain_history':hrot,'best_epoch':bt,'metrics':classification_metrics(yt2,prob2,args.seed)}
    torch.save({'model':{k:v.half() if v.is_floating_point() else v for k,v in backbone.state_dict().items()},'architecture':'resnet18_digits','num_classes':10,'input_size':32,'pretraining':'self-supervised 4-way rotations on train split'},out/'artifacts'/'rotation_resnet18.pt')
    np.savez_compressed(out/'artifacts'/'rotation_resnet18_predictions.npz',y_true=yt2,probabilities=prob2,test_indices=splits['test'])
    print('TRANSFER held-out accuracy:',result['models']['rotation_resnet18']['metrics']['accuracy'],flush=True)
    result['elapsed_sec']=round(time.time()-start,1)
    (out/'results.json').write_text(json.dumps(result,indent=2,default=str))
    print('Saved',out/'results.json', 'seconds',result['elapsed_sec'])

if __name__=='__main__':main()
