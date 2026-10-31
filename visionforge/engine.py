"""Reusable fit/predict loops with validation-only checkpoint selection."""
import copy
import numpy as np
import torch
from torch import nn

def run_epoch(model, loader, optimizer=None, device='cpu'):
    training=optimizer is not None
    model.train(training)
    total=correct=0; loss_sum=0.
    for xb,yb in loader:
        xb=xb.to(device); yb=yb.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits=model(xb)
            loss=nn.functional.cross_entropy(logits,yb)
            if training:
                loss.backward(); optimizer.step()
        total+=yb.shape[0]; correct+=(logits.argmax(1)==yb).sum().item(); loss_sum+=loss.item()*yb.shape[0]
    return {'loss':loss_sum/total,'accuracy':correct/total}

def fit(model,train_loader,val_loader,epochs=8,lr=1e-3,device='cpu',weight_decay=1e-4,tag=''):
    model.to(device)
    opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=lr,weight_decay=weight_decay)
    history=[]; best=-1; best_state=None; best_epoch=0
    for epoch in range(1,epochs+1):
        tr=run_epoch(model,train_loader,opt,device); va=run_epoch(model,val_loader,device=device)
        row={'epoch':epoch,'train_loss':round(tr['loss'],5),'train_accuracy':round(tr['accuracy'],5),'val_loss':round(va['loss'],5),'val_accuracy':round(va['accuracy'],5)}
        history.append(row)
        print(f'{tag:>18} epoch {epoch:02} train={tr["accuracy"]:.3f} val={va["accuracy"]:.3f}',flush=True)
        if va['accuracy']>best:
            best=va['accuracy']; best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return history,best_epoch

def predict(model, loader,device='cpu'):
    model.to(device).eval(); probs=[]; labels=[]
    with torch.inference_mode():
        for xb,yb in loader:
            probs.append(torch.softmax(model(xb.to(device)),dim=1).cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(labels),np.concatenate(probs)

class RotationPretext(torch.utils.data.Dataset):
    """Four-way rotation prediction on the training split only (source task)."""
    def __init__(self,base_dataset): self.base=base_dataset
    def __len__(self): return len(self.base)*4
    def __getitem__(self,index):
        x,_=self.base[index//4]
        rotation=index%4
        return torch.rot90(x,rotation,[1,2]), rotation
