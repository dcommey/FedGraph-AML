import pytest
import torch
from experiments.ablations_routing import source_available


def test_availability_is_nested_and_stable_across_prefix_lengths():
    ids=torch.arange(100)
    a=source_available(ids,.25);b=source_available(ids,.5)
    assert bool((~a|b).all())
    assert torch.equal(a,source_available(torch.arange(1000),.25)[:100])
    assert not source_available(ids,0.).any() and source_available(ids,1.).all()
    if torch.cuda.is_available():
        assert torch.equal(a,source_available(ids.cuda(),.25).cpu())
