"""Create a self-contained research report using real benchmark outputs."""
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs';A=DOC/'assets';S=DOC/'screenshots'
res=json.loads((ROOT/'experiments/offline_digits/results.json').read_text())
path=DOC/'VISIONFORGE_Research_Report.pdf'
for name,p in [('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),('DejaVuBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name,p))
w,h=A4;navy=colors.HexColor('#08151d');panel=colors.HexColor('#122b30');mint=colors.HexColor('#1b8066');mintlight=colors.HexColor('#bdf7da');muted=colors.HexColor('#667a80');ink=colors.HexColor('#13252c');gold=colors.HexColor('#e5a16a');white=colors.HexColor('#FFFFFF')
c=canvas.Canvas(str(path),pagesize=A4);c.setTitle('VISIONFORGE | Explainable Computer Vision Research Report');c.setAuthor('VISIONFORGE independent research portfolio');c.setSubject('Custom CNN, ResNet-18 within-domain transfer, evaluation and Grad-CAM')
normal=ParagraphStyle('p',fontName='DejaVu',fontSize=9.2,leading=15,textColor=ink,spaceAfter=0)
small=ParagraphStyle('s',parent=normal,fontSize=8.1,leading=13)
light=ParagraphStyle('light',parent=normal,textColor=colors.HexColor('#dceee7'),fontSize=11,leading=19)

def para(t,x,y,width,style=normal):
    p=Paragraph(t,style);_,ph=p.wrap(width,700);p.drawOn(c,x,y-ph);return y-ph

def footer(page,section):
    c.setStrokeColor(colors.HexColor('#dce6e3'));c.line(52,48,w-52,48);c.setFont('DejaVu',7);c.setFillColor(muted)
    c.drawString(52,33,'VISIONFORGE / INDEPENDENT AI RESEARCH PORTFOLIO');c.drawRightString(w-52,33,f'{section}  /  {page:02d}')

def heading(chapter,title,subtitle=''):
    c.setFillColor(white);c.rect(0,0,w,h,fill=1,stroke=0)
    c.setFont('DejaVuBold',9);c.setFillColor(mint);c.drawString(52,h-65,chapter.upper())
    c.setFont('DejaVuBold',23);c.setFillColor(ink);c.drawString(52,h-105,title)
    if subtitle:
        c.setFont('DejaVu',9);c.setFillColor(muted);c.drawString(52,h-126,subtitle)
    c.setStrokeColor(colors.HexColor('#cbded7'));c.line(52,h-144,w-52,h-144)

def imagefit(fp,x,y,tw,th):
    im=ImageReader(str(fp));iw,ih=im.getSize();scale=min(tw/iw,th/ih)
    width=iw*scale;height=ih*scale
    c.drawImage(im,x+(tw-width)/2,y+(th-height)/2,width,height,mask='auto',preserveAspectRatio=True)

def bullet(t,y,x=52,width=None):
    c.setFillColor(mint);c.circle(x+3,y-7,2,stroke=0,fill=1)
    return para(t,x+17,y,(width or (w-104))-17,small)-9

# COVER
c.setFillColor(navy);c.rect(0,0,w,h,fill=1,stroke=0)
c.setFillColor(mintlight);c.setFont('DejaVuBold',15);c.drawString(52,h-77,'◈  VISIONFORGE')
c.setFillColor(colors.HexColor('#86a7a4'));c.setFont('DejaVu',8);c.drawString(52,h-96,'PERCEPTION / EXPLANATION / REPRODUCIBILITY')
c.setFillColor(white);c.setFont('DejaVuBold',35);c.drawString(52,h-167,'See the model.')
c.setFillColor(mintlight);c.drawString(52,h-212,'Question the answer.')
para('A controlled comparison of a trained custom CNN and a rotation-pretrained ResNet-18 with real Grad-CAM explanations.',52,h-245,w-105,light)
c.setStrokeColor(mintlight);c.line(52,h-312,w-52,h-312)
imagefit(S/'01_dashboard_hero.png',45,166,w-90,335)
c.setFont('DejaVu',8);c.setFillColor(colors.HexColor('#a2c2b8'));c.drawString(52,129,'RESEARCH REPORT  /  VF-001  /  OCTOBER 2026')
c.drawString(52,109,'Executed PyTorch CPU benchmark  •  2 saved models  •  315 held-out test images')
c.drawString(52,87,'Independent project. Not affiliated with or endorsed by BTU Cottbus-Senftenberg.')
c.showPage()

# 2 - DESIGN
heading('01 / research design','The experiment at a glance','Hypothesis-driven model comparison with fixed partitions')
y=h-172
y=para('Research question',52,y,w-104,ParagraphStyle('head1',parent=normal,fontName='DejaVuBold',fontSize=13,leading=18))-12
y=para('Can a compact convolutional neural network match or outperform a larger ResNet-18 with within-domain rotation pretraining on small image data? Can both approaches be examined using class-specific visual attribution?',52,y,w-104)-26
c.setFillColor(colors.HexColor('#eef6f1'));c.roundRect(52,y-150,w-104,152,7,stroke=0,fill=1)
cols=[('DATASET','1,797 images','scikit-learn handwritten digits'),('TRAIN','1,168 images','supervised + rotation pretext'),('VALIDATION','314 images','best epoch selection'),('TEST','315 images','holdout; same for both models')]
for i,(a,b,d) in enumerate(cols):
    x=70+i*122
    c.setFont('DejaVuBold',8);c.setFillColor(mint);c.drawString(x,y-26,a)
    c.setFont('DejaVuBold',15);c.setFillColor(ink);c.drawString(x,y-57,b)
    para(d,x,y-68,111,ParagraphStyle('tiny'+str(i),parent=small,fontSize=7,leading=10))
y-=196
c.setFont('DejaVuBold',13);c.setFillColor(ink);c.drawString(52,y,'Training routes')
y-=24
for title,txt in [
 ('MODEL A  /  SUPERVISED CNN','Learned 32/64/96 channel convolution stages; random initialization. AdamW, learning rate 0.002, 14 epochs; validation-best at epoch 11.'),
 ('MODEL B  /  RESNET-18 TRANSFER','Pretrain a 1-channel ResNet-18 to recognize 0/90/180/270-degree image rotations using training images only. Replace the head with 10 outputs and fine-tune all layers for 6 epochs.'),
 ('COMMON EVALUATION','Predict exactly once on the same 315-image holdout after validation selection. Report classwise precision/recall/F1, macro OvR ROC-AUC, ECE and bootstrap intervals.')]:
    c.setFillColor(mint);c.setFont('DejaVuBold',8);c.drawString(52,y,title);y-=12
    y=para(txt,52,y,w-106,small)-18
footer(2,'METHODS');c.showPage()

# 3 results
heading('02 / held-out results','What the models actually achieved','Measured on the same 315 test images; numbers from saved predictions')
y=h-180
metrics=[('Accuracy','98.730%','98.095%'),('Macro precision','98.749%','98.286%'),('Macro recall','98.729%','98.083%'),('Macro F1','98.724%','98.120%'),('Macro OvR ROC-AUC','0.999887','0.999932'),('10-bin ECE (lower better)','0.044678','0.009299'),('Correct images','311 / 315','309 / 315')]
c.setFillColor(panel);c.roundRect(52,y-33,w-104,33,5,stroke=0,fill=1)
c.setFillColor(white);c.setFont('DejaVuBold',8);c.drawString(62,y-22,'METRIC');c.drawString(302,y-22,'CUSTOM CNN');c.drawString(412,y-22,'TRANSFER RESNET-18');y-=33
for i,(l,a,b) in enumerate(metrics):
    if i%2==0:c.setFillColor(colors.HexColor('#f2f6f3'));c.rect(52,y-28,w-104,28,fill=1,stroke=0)
    c.setFont('DejaVu',8);c.setFillColor(ink);c.drawString(62,y-18,l);c.drawString(302,y-18,a);c.drawString(415,y-18,b);y-=28
y-=20
imagefit(A/'model_comparison.png',52,y-196,w-104,199);y-=217
c.setFillColor(ink);c.setFont('DejaVuBold',11);c.drawString(52,y,'Interpretation')
y-=11
y=para('The smaller CNN is <b>slightly more accurate</b> (two additional correct test images). The rotation-pretrained ResNet-18 has marginally higher OvR ROC-AUC and a smaller 10-bin ECE. These differences are exploratory: one random split, one seed and unequal training budgets cannot establish a statistically significant architecture advantage.',52,y,w-104,small)
footer(3,'RESULTS');c.showPage()

# 4 figures
heading('03 / evaluation figures','Learning, errors and uncertainty','Best validation checkpoint selected before held-out evaluation')
imagefit(A/'training_curves.png',48,h-405,w-96,250)
c.setFont('DejaVuBold',9);c.setFillColor(ink);c.drawString(52,h-430,'Confusion matrices - 10 digits, 315 samples')
imagefit(A/'confusion_custom_cnn.png',50,h-705,240,265)
imagefit(A/'confusion_rotation_resnet18.png',305,h-705,240,265)
para('See additional calibration and ROC curves under docs/assets/. Confusion matrix cells are true/predicted class counts. The bootstrap 95% accuracy intervals are 97.46-99.68% for CNN and 96.51-99.37% for ResNet-18.',52,113,w-104,small)
footer(4,'DIAGNOSTICS');c.showPage()

# 5 gradcam
heading('04 / model interpretation','Class-specific Grad-CAM','Maps generated from actual gradients of saved checkpoints on held-out samples')
imagefit(A/'gradcam_gallery.png',54,124,w-108,540)
para('The heatmap indicates class-sensitive feature regions. It is not a segmentation mask, proof of causality, or a reliability guarantee. The web workbench allows the reader to switch between both models and inspect 12 held-out cases.',52,114,w-104,small)
footer(5,'EXPLAINABILITY');c.showPage()

# 6 conclude
heading('05 / reproducibility & context','A research workflow, not a claim','Curriculum relevance and project limitations')
y=h-172
c.setFillColor(ink);c.setFont('DejaVuBold',13);c.drawString(52,y,'Package contents');y-=18
for text in [
 'Complete Python/PyTorch source, two finished model checkpoints and source rotation-pretraining weights.',
 'Test predictions with indices, experiment JSON, per-class metrics and all plotted figures.',
 'Interactive static GitHub Pages dashboard, 12 real explanation cases, six actual Chromium screenshots.',
 'Streamlit app for digit image uploads and model inference, plus automated tests and publication workflow.']:
    y=bullet(text,y)
y-=9
c.setFillColor(ink);c.setFont('DejaVuBold',13);c.drawString(52,y,'What this benchmark cannot demonstrate');y-=19
for text in [
 'The source pictures are just 8x8 grayscale digits; upscaling to 32x32 does not create visual detail.',
 'Within-domain rotation pretraining is genuine source-task transfer but it is NOT ImageNet pretraining.',
 'Only one random split/seed was executed, with unequal pretraining compute and 315 held-out images.',
 'The ImageNet-to-CIFAR-10 script is implemented but NOT run; no CIFAR-10 metrics are claimed.',
 'Grad-CAM is an attribution heuristic; its visual plausibility is not evidence of causal explanation.']:
    y=bullet(text,y)
y-=7
c.setFillColor(ink);c.setFont('DejaVuBold',13);c.drawString(52,y,'BTU curriculum alignment');y-=19
for text in [
 '<b>14186 Image Processing and Computer Vision:</b> filters, learned image representations and visual tasks.',
 '<b>14021 Explainable Machine Learning:</b> class attribution and neural visualization.',
 '<b>14449 Workflows for Machine Learning and Reproducible Science Laboratory:</b> experiment design, evaluation, code audit and documentation.']:
    y=bullet(text,y)
y-=14
c.setFillColor(mint);c.setFont('DejaVuBold',8);c.drawString(52,y,'OFFICIAL CURRICULUM SOURCE');y-=13
source='https://www-docs.b-tu.de/studierende/public/modulhandbuch/master/Artificial_Intelligence/SoSe_2026/Modulhandbuch_SS2026_88_ARI_PO2022.pdf'
y=para('BTU AI M.Sc. module handbook, summer semester 2026 (pp. 97, 121, 152). See project docs/BTU_ALIGNMENT.md for full links. Curriculum overlap does not guarantee admission.',52,y,w-104,small)
c.linkURL(source,(52,y-8,w-52,y+26),relative=0)
footer(6,'REPRODUCIBILITY');c.showPage()
c.save();print('Saved',path,path.stat().st_size)
