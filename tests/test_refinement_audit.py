"""Check the independent audit on tied and ordered synthetic evidence."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from experiments.summarize_refinements import average_precision,roc_auc,f1_threshold
from experiments.corrected_evaluation import calibration


def test_independent_ap_agrees_with_sklearn_on_ties():
    rng=np.random.default_rng(70)
    for n in [10,100,1000]:
        y=rng.integers(0,2,n);y[:2]=[0,1]
        for p in [rng.random(n),rng.integers(0,5,n)/5,np.ones(n)*.5]:
            assert abs(average_precision(y,p)-average_precision_score(y,p))<1e-12
            assert abs(roc_auc(y,p)-roc_auc_score(y,p))<1e-12
            assert f1_threshold(y,p)==calibration(y,p)
