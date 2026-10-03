"""Small causal/provenance checks on the independent transaction adapter."""
import numpy as np
import pandas as pd
import torch
from data.refinement_data import load_ibm,ibm_sample_keep,continuity_edges


def test_simultaneous_payments_do_not_evict_strictly_earlier_history():
    seconds=np.array([1,2,3,4,4,4,4,5],dtype=np.int64)
    accounts=['account']*len(seconds)
    edges=continuity_edges(seconds,accounts,accounts)
    for destination in range(3,7):
        assert np.array_equal(edges[0,edges[1]==destination],[0,1,2])
    assert np.array_equal(edges[0,edges[1]==7],[4,5,6])
    assert bool((seconds[edges[0]]<seconds[edges[1]]).all())


def test_ibm_adapter_excludes_tail_labels_and_same_time_edges(tmp_path):
    header=['Timestamp','From Bank','Account','To Bank','Account','Amount Received',
            'Receiving Currency','Amount Paid','Payment Currency','Payment Format','Is Laundering']
    rows=[]
    for i in range(200):
        day=1+i%10
        # Every source/target same account; simultaneous records are not ordered.
        rows.append([f'2022/09/{day:02d} 12:00',1,'000A',1,'000A',10+i,'USD',10+i,'USD','Wire',i%2])
    rows += [['2022/09/12 12:00',1,'000A',1,'000A',5,'USD',5,'USD','Wire',1]]
    raw=tmp_path/'transactions.csv'
    pd.DataFrame(rows,columns=header).to_csv(raw,index=False)
    d,meta=load_ibm(raw,tmp_path/'cache')
    assert bool((d.timestep>=0).all() and (d.timestep<=9).all())
    assert meta['excluded_outside_primary_period']==1
    assert np.array_equal(np.sort(d.transaction_ids.numpy()),np.where(ibm_sample_keep(np.arange(200)))[0])
    # No same-timestamp edges; daily edges only when24h condition is satisfied.
    assert bool((d.timestep[d.edge_index[0]]<d.timestep[d.edge_index[1]]).all())
    changed=tmp_path/'changed.csv'
    rows2=[r[:-1]+[1-r[-1]] for r in rows]
    pd.DataFrame(rows2,columns=header).to_csv(changed,index=False)
    e,_=load_ibm(changed,tmp_path/'cache2')
    torch.testing.assert_close(d.x,e.x)
    torch.testing.assert_close(d.edge_index,e.edge_index)
    torch.testing.assert_close(d.transaction_ids,e.transaction_ids)
