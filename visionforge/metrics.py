"""Held-out model metrics, calibration, confidence intervals and saved predictions."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

def classification_metrics(y_true, prob, seed=42):
    y_true=np.asarray(y_true,dtype=int); prob=np.asarray(prob)
    y_pred=prob.argmax(axis=1)
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,labels=list(range(prob.shape[1])),average='macro',zero_division=0)
    pp,rr,ff,sup=precision_recall_fscore_support(y_true,y_pred,labels=list(range(prob.shape[1])),average=None,zero_division=0)
    confidence=prob.max(1); correctness=(y_pred==y_true).astype(float)
    ece=0.
    for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(confidence>a)&(confidence<=b)
        if mask.any(): ece+=mask.mean()*abs(confidence[mask].mean()-correctness[mask].mean())
    rng=np.random.default_rng(seed)
    boots=[]
    for _ in range(400):
        ii=rng.integers(0,len(y_true),len(y_true))
        boots.append(float((y_pred[ii]==y_true[ii]).mean()))
    try: auc=float(roc_auc_score(y_true,prob,multi_class='ovr',average='macro',labels=list(range(prob.shape[1]))))
    except ValueError: auc=None
    return dict(n_test=len(y_true),accuracy=float(accuracy_score(y_true,y_pred)),accuracy_ci95=[float(v) for v in np.quantile(boots,[.025,.975])],
                precision_macro=float(p),recall_macro=float(r),f1_macro=float(f),roc_auc_ovr_macro=auc,expected_calibration_error_10bin=float(ece),
                confusion_matrix=confusion_matrix(y_true,y_pred,labels=list(range(prob.shape[1]))).tolist(),
                per_class=[dict(label=str(i),support=int(sup[i]),precision=float(pp[i]),recall=float(rr[i]),f1=float(ff[i])) for i in range(prob.shape[1])])
