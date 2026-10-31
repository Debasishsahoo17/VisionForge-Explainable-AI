"""Live inference app. Deployment: Streamlit Community Cloud -> app.py.
This app uses the INCLUDED offline digit checkpoints; no local training needed.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageOps
import streamlit as st
from sklearn.datasets import load_digits
from visionforge.models import CustomCNN,make_resnet18
from visionforge.explain import gradcam

ROOT=Path(__file__).parent
st.set_page_config(page_title='VISIONFORGE / Neural Explainability Lab',page_icon='◈',layout='wide')
st.markdown('''<style>
.stApp{background:#0b1120;color:#e9efff}
h1,h2,h3{letter-spacing:-.04em!important}
[data-testid="stMetric"]{background:#111e30;padding:18px;border:1px solid #2b3c56;border-radius:16px}
</style>''',unsafe_allow_html=True)
@st.cache_resource
def get_models():
    paths=[ROOT/'experiments/offline_digits/artifacts/custom_cnn.pt',ROOT/'experiments/offline_digits/artifacts/rotation_resnet18.pt']
    models=[CustomCNN(),make_resnet18(10,mode='digits')]
    for model,path in zip(models,paths):
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        model.load_state_dict(ckpt['model']);model.eval()
    return models

st.caption('VISIONFORGE / RESEARCH WORKSPACE / OFFLINE DIGITS BENCHMARK')
st.title('Understand the prediction, not just the label.')
st.write('Inspect a real trained CNN and rotation-pretrained ResNet-18 using class-specific Grad-CAM explanations. Both models were evaluated on a held-out subset of scikit-learn digits.')
models=get_models()
left,right=st.columns([1,2],gap='large')
with left:
    input_type=st.radio('Input source',['Benchmark sample','Upload a handwritten digit'],horizontal=True)
    if input_type=='Benchmark sample':
        ix=st.slider('Benchmark sample index',0,1796,125)
        record=load_digits(); image=Image.fromarray(np.uint8(record.images[ix]*255/16),'L')
        st.caption(f'Ground-truth digit: {record.target[ix]} (sample may be from training, validation or test)')
    else:
        upload=st.file_uploader('Upload grayscale/RGB handwriting image',type=['png','jpg','jpeg'])
        invert=st.checkbox('Invert black ink on white paper',value=True)
        if not upload: st.info('Upload a single handwritten digit to continue.');st.stop()
        image=Image.open(upload).convert('L')
        if invert:image=ImageOps.invert(image)
        st.caption('Out-of-distribution uploads may produce unreliable predictions; the models were trained on a small 8×8 digit benchmark.')
    image=image.resize((32,32),Image.Resampling.BILINEAR)
    st.image(image,width=224,caption='Model input • 32 × 32 grayscale')
    selected=st.selectbox('Explanation target',['Predicted class','Digit 0','Digit 1','Digit 2','Digit 3','Digit 4','Digit 5','Digit 6','Digit 7','Digit 8','Digit 9'])
with right:
    src=np.array(image,dtype=np.float32)/255.
    x=torch.tensor(src)[None]
    cols=st.columns(2,gap='large')
    for c,model,name in zip(cols,models,['Custom CNN','Rotation-pretrained ResNet-18']):
        class_id=None if selected=='Predicted class' else int(selected.split()[-1])
        cam,pred,conf=gradcam(model,x,target_class=class_id)
        with torch.no_grad(): probs=torch.softmax(model(x.unsqueeze(0)),1)[0].numpy()
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(5,5));ax.imshow(src,cmap='gray',vmin=0,vmax=1)
        ax.imshow(cam,cmap='turbo',alpha=.55,vmin=0,vmax=1);ax.axis('off');fig.tight_layout(pad=0)
        with c:
            st.subheader(name)
            st.metric('Predicted digit',str(pred),f'{conf:.1%} confidence')
            st.pyplot(fig,clear_figure=True)
            st.caption('Grad-CAM highlights model-sensitive regions; these are not causal proof or segmentation masks.')
            st.bar_chart({str(i):float(p) for i,p in enumerate(probs)},horizontal=True)
st.divider()
st.caption('Experimental evidence and complete audit trail: experiments/offline_digits/results.json. Self-supervised rotation pretraining is not ImageNet transfer.')
