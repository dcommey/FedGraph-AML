"""Additional invariants without mutating the hashed numerical-test source."""
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from torch_geometric.data import Data
from experiments.refinement_diagnostic import routed_exchange
from tests.test_corrected_protocol import fixture_data
from experiments.corrected_evaluation import build_views
from models.refined_sage import RefinedSAGE


class IdentityState:
    norm=SimpleNamespace(normalized_shape=[2])
    def eval(self):return self
    def first_layer(self,x,edge,raw=None):return x


def test_shuffled_sources_consistent_across_edges_and_recipients():
    def view(ids,src,times):
        return dict(ids=torch.tensor(ids),x=torch.tensor([[float(i),float(times[j])] for j,i in enumerate(ids)]),
            edge_index=torch.empty((2,0),dtype=torch.long),remote_src=torch.tensor(src,dtype=torch.long),
            remote_dst=torch.zeros(len(src),dtype=torch.long),timestep=torch.tensor(times))
    views=[view([0,1,2],[],[0,0,1]),view([3],[0,0,1,2],[0]),view([4],[0,1,2],[0])]
    for seed in range(10):
        remote,_,comm=routed_exchange([IdentityState()]*3,views,'cpu','shuffled',seed)
        first,second=remote[1][1],remote[2][1]
        torch.testing.assert_close(first[0],first[1])
        torch.testing.assert_close(first[[0,2,3]],second)
        assert first[0,0] in [0,1] and first[2,0] in [0,1]
        torch.testing.assert_close(first[3],torch.tensor([2.,1.]))
        assert comm['unique_sender_bytes']==3*2*4 and comm['receiver_fanout_bytes']==6*2*4


def test_zero_message_forward_and_detached_gradient():
    data=fixture_data();owner=torch.tensor([0,1,0,1]);data.timestep[:]=0
    views=build_views(data,owner,2,0)
    torch.manual_seed(55);models=[RefinedSAGE(2,8,0) for _ in views]
    remote,_,_=routed_exchange(models,views,'cpu',fraction=0)
    for m,v,r in zip(models,views,remote):
        torch.testing.assert_close(m(v['x'],v['edge_index'],r),m(v['x'],v['edge_index']))
    with_messages,_,_=routed_exchange(models,views,'cpu')
    for r in with_messages:
        if r is not None:assert not r[1].requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(),reason='runner memory instrumentation requires CUDA')
def test_current_fresh_one_step_training_parity(tmp_path):
    from experiments.refinement_study import run_neural
    ids=torch.arange(16);step=torch.cat([torch.zeros(8),torch.ones(4),torch.ones(4)*2]).long()
    data=Data(x=torch.stack([ids.float()/16,torch.cos(ids.float())],1),y=ids%2,timestep=step,
        edge_index=torch.tensor([[0,1,2,4,5,6,8,10,12,14],[1,2,3,5,6,7,9,11,13,15]]))
    data.train_end=0;data.val_steps=[1,2];data.test_steps=[];data.no_cross_time_edges=True
    data.train_mask=step==0;data.val_mask=step>0;owner=(ids//2)%2
    args=SimpleNamespace(clients=2,device='cuda',rounds=2,local_epochs=1,evaluate_every=1,fraction=1.,dataset='tiny')
    config=dict(lr=.003,hidden=8,dropout=0.,reset_adam=False,weight_decay=.0005)
    for method in ['current','fresh']:
        run_neural(data,owner,1,42,'fixture',method,config,args,tmp_path/method,'pilot')
    a=np.load(tmp_path/'current'/'validation_predictions.npz');b=np.load(tmp_path/'fresh'/'validation_predictions.npz')
    np.testing.assert_allclose(a['probability'],b['probability'],atol=1e-7,rtol=1e-6)
    left=torch.load(tmp_path/'current'/'checkpoint.pt',weights_only=False);right=torch.load(tmp_path/'fresh'/'checkpoint.pt',weights_only=False)
    assert left['selected_round']==right['selected_round']
    for ls,rs in zip(left['states'],right['states']):
        for name in ls:torch.testing.assert_close(ls[name],rs[name],atol=1e-7,rtol=1e-6)
