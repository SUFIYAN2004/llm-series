import pytest
import torch

from llmseries.config import ALL, V
from llmseries.models import make
from llmseries.train import aux_loss

NEURAL = [n for n in ALL if n != "ngram"]


@pytest.mark.parametrize("name", NEURAL)
def test_forward_backward(name):
    m = make(name, 64)
    out = m(torch.randint(0, V, (2, 64)))
    assert out.shape == (2, 64, V)
    (out.float().mean() + aux_loss(m)).backward()


@pytest.mark.parametrize("name", NEURAL)
def test_causality(name):
    """Changing the last token must not change any earlier logit (no future leakage)."""
    torch.manual_seed(0)
    m = make(name, 64).eval()
    x = torch.randint(0, V, (2, 64)); y = x.clone(); y[:, -1] = (y[:, -1] + 1) % V
    with torch.no_grad():
        a, b = m(x), m(y)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-4), f"{name} leaks future information"
