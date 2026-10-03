"""Scientific invariants for the proposed exchange controls."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.corrected_evaluation import build_views
from experiments.refinement_diagnostic import routed_exchange, validation_predict
from models.refined_sage import RefinedSAGE
from tests.test_corrected_protocol import fixture_data


def test_oracle_frozen_forward_equals_complete_graph():
    data = fixture_data()
    data.timestep[:] = 0
    owner = torch.tensor([0, 1, 0, 1])
    views = build_views(data, owner, 2, 0)
    torch.manual_seed(7)
    base = RefinedSAGE(2, 8, 0).eval()
    models = [copy.deepcopy(base), copy.deepcopy(base)]
    remote, raw, comm = routed_exchange(models, views, 'cpu', 'oracle')
    out = torch.empty(data.num_nodes)
    for model, view, foreign, first in zip(models, views, remote, raw):
        out[view['ids']] = model(view['x'], view['edge_index'], foreign, first)
    torch.testing.assert_close(out, base(data.x, data.edge_index), atol=1e-6, rtol=1e-5)
    assert comm['oracle_raw_fanout_bytes'] > 0


def test_zero_fraction_and_shuffled_routing():
    data = fixture_data()
    data.timestep[:] = 0
    views = build_views(data, torch.tensor([0, 1, 0, 1]), 2, 0)
    models = [RefinedSAGE(2, 8, 0).eval() for _ in views]
    remote, raw, comm = routed_exchange(models, views, 'cpu', fraction=0)
    assert remote == [None, None] and raw == [None, None]
    assert comm['received_edges'] == 0
    original = routed_exchange(models, views, 'cpu')[0]
    shuffled = routed_exchange(models, views, 'cpu', 'shuffled', seed=3)[0]
    for a, b in zip(original, shuffled):
        if a is not None:
            assert torch.equal(a[0], b[0])
            assert a[1].shape == b[1].shape
    with pytest.raises(ValueError, match='fraction=1'):
        routed_exchange(models, views, 'cpu', 'oracle', fraction=0)


def test_vectorized_current_parity_with_previous_exchange():
    from experiments.corrected_evaluation import exchange
    data = fixture_data()
    data.timestep[:] = 0
    views = build_views(data, torch.tensor([0, 1, 0, 1]), 2, 0)
    models = [RefinedSAGE(2, 8, 0).eval() for _ in views]
    old, old_comm = exchange(models, views, 'cpu')
    new, _, comm = routed_exchange(models, views, 'cpu')
    for a, b in zip(old, new):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a[0], b[0])
            torch.testing.assert_close(a[1], b[1])
    assert old_comm['transmitted_bytes'] == comm['unique_sender_bytes']


def test_diagnostic_has_no_test_evaluation():
    import inspect
    import experiments.refinement_diagnostic as module
    source = inspect.getsource(module)
    assert "split == 'test'" not in source
    assert "data.test_mask" not in source
    assert "test_predictions.npz" not in source
