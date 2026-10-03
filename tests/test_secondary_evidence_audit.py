"""Verify exposure auditing accepts known evidence and rejects corruption."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.summarize_refinement_secondary import audit_exposure


class ExposureAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        truth=np.array([[0,1],[1,0],[-1,2],[2,-1]],dtype=np.float32)
        arrays=dict(auxiliary_ids=np.array([10,11]),validation_ids=np.array([12]),
            evaluation_ids=np.array([13,14,15,16]),truth_normalized_features=truth,
            mean=np.zeros_like(truth),ridge=truth+.1,nonlinear_mlp=truth*.8)
        np.savez(self.root/'reconstruction_exposure.npz',**arrays)
        rows=[]
        for name in ['mean','ridge','nonlinear_mlp']:
            mse=((truth-arrays[name])**2).mean(0)
            rows.append(dict(attack=name,evaluation=dict(n=4,per_feature_mse=mse.tolist(),
                mean_normalized_feature_mse=float(mse.mean()),rmse=float(np.sqrt(mse.mean())),
                mean_feature_r2=float(np.mean(1-mse/truth.var(0))))))
        rows[1].update(selected_alpha=1,validation_trials=[
            dict(alpha=.1,validation=dict(mean_normalized_feature_mse=.2)),
            dict(alpha=1,validation=dict(mean_normalized_feature_mse=.1))])
        self.membership=dict(auxiliary_ids=np.arange(4),evaluation_ids=np.arange(4,8),
            auxiliary_membership=np.array([0,1,0,1]),membership=np.array([0,1,0,1]),
            logistic=np.array([.1,.8,.2,.9]),random_forest=np.array([.8,.1,.6,.2]))
        for split in ['auxiliary','evaluation']:
            for field in ['timestep','owner','label','degree_bucket']:
                self.membership[split+'_'+field]=np.zeros(4,dtype=int)
        np.savez(self.root/'membership_exposure.npz',**self.membership)
        self.record=dict(reconstruction=rows,supervised_membership=[
            dict(attack=method,roc_auc=auc,tpr_at_fpr_001=auc,n_auxiliary=4,n_evaluation=4,
                auxiliary_member_count=2,evaluation_member_count=2)
            for method,auc in [('logistic',1.),('random_forest',0.)]])

    def test_known_scores_and_errors_pass(self):
        self.assertEqual(audit_exposure(self.root,self.record)['status'],'passed')

    def test_changed_attack_score_rejected(self):
        self.membership['logistic']=1-self.membership['logistic']
        np.savez(self.root/'membership_exposure.npz',**self.membership)
        with self.assertRaisesRegex(AssertionError,'operating evidence'):
            audit_exposure(self.root,self.record)

    def test_split_overlap_rejected(self):
        self.membership['evaluation_ids'][0]=0
        np.savez(self.root/'membership_exposure.npz',**self.membership)
        with self.assertRaisesRegex(AssertionError,'splits overlap'):
            audit_exposure(self.root,self.record)

    def test_stratum_imbalance_rejected(self):
        self.membership['auxiliary_membership'][0]=1
        np.savez(self.root/'membership_exposure.npz',**self.membership)
        with self.assertRaisesRegex(AssertionError,'not balanced'):
            audit_exposure(self.root,self.record)


if __name__=='__main__':unittest.main()
