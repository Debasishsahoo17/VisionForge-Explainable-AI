#!/usr/bin/env python3
"""Generate real graphs, Grad-CAM images and GitHub Pages data from trained artifacts."""
import json, sys, shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cv2
from sklearn.metrics import roc_curve,auc
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from visionforge.models import CustomCNN,make_resnet18
from visionforge.explain import gradcam
from visionforge.data import load_digit_splits

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'experiments/offline_digits'
D=ROOT/'docs/assets'
W=ROOT/'web/assets'
for p in [D,W,W/'samples']:p.mkdir(parents=True,exist_ok=True)
result=json.loads((E/'results.json').read_text())
shutil.copyfile(E/'results.json',W/'results.json')
np1=np.load(E/'artifacts/custom_cnn_predictions.npz')
np2=np.load(E/'artifacts/rotation_resnet18_predictions.npz')
ys=np1['y_true']; p1=np1['probabilities'];p2=np2['probabilities'];test_indices=np1['test_indices']
assert np.array_equal(ys,np2['y_true']) and np.array_equal(test_indices,np2['test_indices'])
loaders,X,Y,inds=load_digit_splits()
assert np.array_equal(inds['test'],test_indices)

plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','figure.facecolor':'#0b1720','axes.facecolor':'#101f2a','axes.edgecolor':'#41535d','axes.labelcolor':'#b4ccc8','xtick.color':'#91aaa9','ytick.color':'#91aaa9','text.color':'#edf9ef','savefig.facecolor':'#0b1720','grid.color':'#294248'})
GREEN='#9eeac7'; GOLD='#f1bd83'; BLUE='#93aef5';
def finish(name,fig):
    fig.savefig(D/name,dpi=165,bbox_inches='tight',pad_inches=.23)
    plt.close(fig)

# Learning curves
fig,axs=plt.subplots(1,2,figsize=(11.5,3.5))
for id_,clr,label in [('custom_cnn',GOLD,'Custom CNN'),('rotation_resnet18',GREEN,'ResNet-18 / rotation transfer')]:
    hist=result['models'][id_]['history']; ep=[r['epoch'] for r in hist]
    axs[0].plot(ep,[r['train_accuracy'] for r in hist],color=clr,ls='--',alpha=.5)
    axs[0].plot(ep,[r['val_accuracy'] for r in hist],color=clr,linewidth=2.3,label=label)
    axs[1].plot(ep,[r['val_loss'] for r in hist],color=clr,linewidth=2.2,label=label)
for ax in axs:ax.grid(alpha=.4);ax.spines[['top','right']].set_visible(False);ax.set_xlabel('Epoch')
axs[0].set_title('Train (dashed) vs validation accuracy');axs[0].set_ylabel('Accuracy');axs[0].legend(frameon=False,fontsize=8)
axs[1].set_title('Validation cross-entropy loss');axs[1].set_ylabel('Loss')
fig.suptitle('Training and validation dynamics',fontsize=15,fontweight='bold');fig.tight_layout();finish('training_curves.png',fig)

# Compare metrics
fig,ax=plt.subplots(figsize=(9.8,4.3))
names=['Accuracy','Precision (macro)','Recall (macro)','F1 (macro)','ROC-AUC (OvR)']
keys=['accuracy','precision_macro','recall_macro','f1_macro','roc_auc_ovr_macro']
m1=result['models']['custom_cnn']['metrics'];m2=result['models']['rotation_resnet18']['metrics']
x=np.arange(len(names));w=.36
bars=ax.bar(x-w/2,[m1[k] for k in keys],w,label='Custom CNN',color=GOLD)
bars2=ax.bar(x+w/2,[m2[k] for k in keys],w,label='Transfer ResNet-18',color=GREEN)
for b in list(bars)+list(bars2):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.006,f'{b.get_height():.3f}',ha='center',fontsize=8)
ax.set_xticks(x,names,rotation=10,ha='right');ax.set_ylim(0,1.13);ax.set_ylabel('Score (0–1)');ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',alpha=.25)
fig.suptitle('Performance on the held-out test partition',fontsize=15,fontweight='bold');fig.tight_layout();finish('model_comparison.png',fig)

for id_,name in [('custom_cnn','confusion_custom_cnn.png'),('rotation_resnet18','confusion_rotation_resnet18.png')]:
    cm=np.array(result['models'][id_]['metrics']['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(6,5.5));im=ax.imshow(cm,cmap='YlGn',vmin=0,vmax=cm.max())
    for i in range(10):
        for j in range(10):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='#08201a' if cm[i,j]>cm.max()*.25 else '#c8d2d3')
    ax.set_xticks(np.arange(10));ax.set_yticks(np.arange(10));ax.set_xlabel('Predicted label');ax.set_ylabel('True label');ax.set_title(result['models'][id_]['display_name'],fontsize=13)
    fig.colorbar(im,ax=ax,shrink=.8,label='Number of examples');fig.tight_layout();finish(name,fig)

# Per-class and ROC curves (genuine multiclass probabilities)
fig,axs=plt.subplots(1,2,figsize=(11,4.4))
for prob,color,label in [(p1,GOLD,'Custom CNN'),(p2,GREEN,'Transfer ResNet-18')]:
    fprs=[];tprs=[];grid=np.linspace(0,1,201)
    for k in range(10):
        fpr,tpr,_=roc_curve((ys==k).astype(int),prob[:,k]);tprs.append(np.interp(grid,fpr,tpr))
    mean_tpr=np.mean(tprs,axis=0)
    axs[0].plot(grid,mean_tpr,color=color,lw=2,label=f'{label} (macro OvR AUC {auc(grid,mean_tpr):.3f})')
axs[0].plot([0,1],[0,1],color='#6c8d92',ls='--',alpha=.6);axs[0].set_xlabel('False positive rate');axs[0].set_ylabel('True positive rate');axs[0].set_title('Macro-averaged one-vs-rest ROC');axs[0].legend(frameon=False,fontsize=8);axs[0].grid(alpha=.3)
x=np.arange(10);w=.37
axs[1].bar(x-w/2,[r['f1'] for r in m1['per_class']],w,color=GOLD,label='Custom CNN')
axs[1].bar(x+w/2,[r['f1'] for r in m2['per_class']],w,color=GREEN,label='Transfer ResNet-18')
axs[1].set_xticks(x);axs[1].set_ylim(0,1.13);axs[1].set_title('F1 scores by digit');axs[1].set_xlabel('Digit');axs[1].set_ylabel('F1');axs[1].legend(frameon=False,fontsize=8)
fig.tight_layout();finish('roc_classwise.png',fig)

# Reliability diagrams, with test leakage avoided in training; evaluation-only.
fig,ax=plt.subplots(figsize=(7,4.6))
bins=np.linspace(0,1,11)
for prob,color,label in [(p1,GOLD,'Custom CNN'),(p2,GREEN,'Transfer ResNet-18')]:
    pred=prob.argmax(1);con=prob.max(1)
    xs=[];ys2=[]
    for a,b in zip(bins[:-1],bins[1:]):
        hit=(con>a)&(con<=b)
        if hit.any():xs.append(float(con[hit].mean()));ys2.append(float((pred[hit]==ys[hit]).mean()))
    ax.plot(xs,ys2,'-o',color=color,label=label,lw=2,ms=5)
ax.plot([0,1],[0,1],ls='--',color='#a6b2c3',label='Perfect calibration');ax.set_xlim(0,1);ax.set_ylim(0,1.05);ax.set_xlabel('Mean confidence');ax.set_ylabel('Observed accuracy');ax.set_title('Reliability (10-bin evaluation)');ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.3);fig.tight_layout();finish('calibration.png',fig)

# Real Grad-CAM predictions and class-sensitive maps.
torch.set_num_threads(2)
cnn=CustomCNN();cnn.load_state_dict(torch.load(E/'artifacts/custom_cnn.pt',weights_only=True,map_location='cpu')['model']);cnn.eval()
resnet=make_resnet18(10,mode='digits');resnet.load_state_dict(torch.load(E/'artifacts/rotation_resnet18.pt',weights_only=True,map_location='cpu')['model']);resnet.eval()
mis1=np.flatnonzero(p1.argmax(1)!=ys);mis2=np.flatnonzero(p2.argmax(1)!=ys)
# Diverse classes, front-load identifiable interesting examples.
chosen=[]
for digit in [8,3,5,2,9,0,6,1,4,7]:
    idx=np.flatnonzero((ys==digit)&(p1.argmax(1)==ys)&(p2.argmax(1)==ys))
    if len(idx):chosen.append(int(idx[np.argmax(np.minimum(p1[idx].max(1),p2[idx].max(1)))]))
for i in list(mis2)+list(mis1):
    if int(i) not in chosen: chosen.append(int(i))
chosen=chosen[:12]

def save_png_from_digit(tensor,path):
    image=(tensor.squeeze().numpy()*255).clip(0,255).astype('uint8')
    Image.fromarray(image,'L').resize((360,360),Image.Resampling.NEAREST).convert('RGB').save(path,optimize=True)
def save_cam(tensor,cam,path):
    base=(tensor.squeeze().numpy()*255).clip(0,255).astype('uint8')
    base=cv2.cvtColor(cv2.resize(base,(360,360),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2RGB)
    heat=cv2.applyColorMap(cv2.resize(np.uint8(cam*255),(360,360),interpolation=cv2.INTER_CUBIC),cv2.COLORMAP_TURBO)
    heat=cv2.cvtColor(heat,cv2.COLOR_BGR2RGB)
    blend=cv2.addWeighted(base,.48,heat,.52,0)
    Image.fromarray(blend).save(path,optimize=True)

showcase=[]
for rank,tidx in enumerate(chosen):
    absid=int(test_indices[tidx]);tensor=X[absid];true=int(ys[tidx]);stem=f'{rank:02d}_digit_{true}'
    original=W/'samples'/f'{stem}_input.png';cnnimg=W/'samples'/f'{stem}_cnn.png';rnimg=W/'samples'/f'{stem}_resnet.png'
    save_png_from_digit(tensor,original)
    hc,_,_=gradcam(cnn,tensor)
    hr,_,_=gradcam(resnet,tensor)
    save_cam(tensor,hc,cnnimg);save_cam(tensor,hr,rnimg)
    obj={'id':rank+1,'dataset_index':absid,'truth':true,'original_image':'assets/samples/'+original.name,
          'cnn_image':'assets/samples/'+cnnimg.name,'resnet_image':'assets/samples/'+rnimg.name,
          'cnn':{'prediction':int(p1[tidx].argmax()),'confidence':float(p1[tidx].max()),'probabilities':p1[tidx].tolist()},
          'resnet':{'prediction':int(p2[tidx].argmax()),'confidence':float(p2[tidx].max()),'probabilities':p2[tidx].tolist()}}
    showcase.append(obj)
(W/'showcase.json').write_text(json.dumps(showcase,indent=2))
shutil.copy(W/'samples'/Path(showcase[0]['resnet_image']).name,W/'hero-attribution.png')
# Portfolio gallery: each sample has real input, CNN CAM, and ResNet CAM.
fig,axes=plt.subplots(4,3,figsize=(9.7,11))
for row,item in enumerate(showcase[:4]):
    for col,(title,path) in enumerate([('Original input',item['original_image']),('Custom CNN CAM',item['cnn_image']),('ResNet-18 CAM',item['resnet_image'])]):
        im=Image.open(W/path.replace('assets/',''))
        ax=axes[row,col];ax.imshow(im);ax.axis('off')
        if row==0:ax.set_title(title,color=["#d5e9e6",GOLD,GREEN][col],fontsize=12,pad=11)
        if col==0:ax.text(-.10,.5,f"True digit {item['truth']}",transform=ax.transAxes,rotation=90,va='center',ha='center',fontsize=10,color='#a7d4c1')
fig.suptitle('Genuine Grad-CAM attributions on held-out images',fontsize=16,y=.99)
fig.tight_layout(pad=2.1);finish('gradcam_gallery.png',fig)

# A compact original grid of 10 original dataset samples.
fig,axs=plt.subplots(2,5,figsize=(9,3.9))
for d,ax in enumerate(axs.flat):
    xx=X[int(np.flatnonzero(Y==d)[0])].squeeze().numpy()
    ax.imshow(xx,cmap='gray');ax.set_title(str(d),fontsize=11);ax.axis('off')
fig.suptitle('Digits benchmark • all 10 classes',fontsize=15);fig.tight_layout();finish('dataset_overview.png',fig)

print('Dashboard examples',len(showcase),'first true label',showcase[0]['truth'],'saved docs figures',len(list(D.glob('*.png'))))
