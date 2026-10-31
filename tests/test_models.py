import numpy as np
import torch
from visionforge.models import CustomCNN,make_resnet18
from visionforge.explain import gradcam
from visionforge.data import load_digit_splits
from visionforge.metrics import classification_metrics

def test_cnn_logits_shape():
    m=CustomCNN();assert tuple(m(torch.rand(2,1,32,32)).shape)==(2,10)

def test_resnet_logits_shape():
    m=make_resnet18();assert tuple(m(torch.rand(2,1,32,32)).shape)==(2,10)

def test_gradcam_output_and_hooks_removed():
    m=CustomCNN()
    layer=m.cam_layer
    before=(len(layer._forward_hooks),len(layer._backward_hooks))
    heat,pred,conf=gradcam(m,torch.rand(1,32,32))
    assert heat.shape==(32,32) and np.all(np.isfinite(heat))
    assert 0<=heat.min()<=heat.max()<=1 and 0<=pred<10 and 0<=conf<=1
    assert before==(len(layer._forward_hooks),len(layer._backward_hooks))

def test_stratified_splits_disjoint():
    _,_,_,ids=load_digit_splits()
    assert len(set(map(int,ids['train'])))+len(set(map(int,ids['val'])))+len(set(map(int,ids['test'])))==1797

def test_metrics_ovr_shape():
    y=np.repeat(np.arange(10),2);prob=np.eye(10)[y]*.91+.009
    met=classification_metrics(y,prob)
    assert met['f1_macro']==1 and met['roc_auc_ovr_macro']==1
    assert len(met['confusion_matrix'])==10

def test_imagenet_model_shape_without_download():
    m=make_resnet18(num_classes=10,mode='imagenet',pretrained=False).eval()
    with torch.no_grad():out=m(torch.rand(1,3,96,96))
    assert tuple(out.shape)==(1,10)
