"""Gradient-weighted Class Activation Mapping (Grad-CAM; Selvaraju et al.)."""
import numpy as np
import torch
import torch.nn.functional as F
from .models import get_cam_layer

def gradcam(model, image, target_class=None):
    """Return nonnegative normalized attribution map, prediction, confidence.

    Hooks are removed in a finally block. The map is class-specific and
    explanatory, *not* a calibrated probability or a segmentation mask.
    """
    was_training=model.training
    model.eval()
    activation={}; gradients={}
    layer=get_cam_layer(model)
    def fw(_mod, _inp, out): activation['v']=out.detach()
    def bw(_mod, _gin, gout): gradients['v']=gout[0].detach()
    h1=layer.register_forward_hook(fw)
    h2=layer.register_full_backward_hook(bw)
    try:
        with torch.enable_grad():
            tensor=image.detach().clone().unsqueeze(0).requires_grad_(True)
            scores=model(tensor)
            probs=torch.softmax(scores,dim=1)
            prediction=int(probs.argmax(1).item())
            target=prediction if target_class is None else int(target_class)
            model.zero_grad(set_to_none=True)
            scores[:,target].sum().backward()
            act=activation['v']; grads=gradients['v']
            weights=grads.mean(dim=(2,3),keepdim=True)
            cam=(weights*act).sum(1,keepdim=True).relu()
            cam=F.interpolate(cam,size=image.shape[-2:],mode='bilinear',align_corners=False)
            cam=cam.squeeze().cpu().numpy()
            cam=np.nan_to_num(cam)
            cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
            return cam, prediction, float(probs[0,prediction].item())
    finally:
        h1.remove(); h2.remove()
        model.train(was_training)
