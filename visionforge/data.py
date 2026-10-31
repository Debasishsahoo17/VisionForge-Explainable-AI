"""Offline, public-domain digit benchmark and leakage-safe stratified splits."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

CLASS_NAMES=[str(i) for i in range(10)]

def load_digit_splits(seed=42, image_size=32, batch_size=64):
    digits=load_digits()
    x=torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1)/16.0
    x=F.interpolate(x,size=(image_size,image_size),mode='bilinear',align_corners=False)
    y=np.array(digits.target, dtype=np.int64)
    idx=np.arange(len(y))
    train, hold=train_test_split(idx,test_size=0.35,stratify=y,random_state=seed)
    val,test=train_test_split(hold,test_size=.5,stratify=y[hold],random_state=seed)
    splits={'train':train,'val':val,'test':test}
    assert not (set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    loaders={}
    for name,ids in splits.items():
        ds=TensorDataset(x[ids],torch.tensor(y[ids]))
        gen=torch.Generator().manual_seed(seed)
        loaders[name]=DataLoader(ds,batch_size=batch_size,shuffle=name=='train',generator=gen,num_workers=0)
    return loaders,x,y,splits
