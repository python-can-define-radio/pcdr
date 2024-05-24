from itertools import count
from datetime import timedelta

from hypothesis import given, settings
from hypothesis.strategies import integers
import numpy as np
import pytest

from pcdr._internal.misc import blockify
from pcdr.unstable.misc2 import first_n_match


@pytest.mark.slow
@given(integers(0, 100_000))
@settings(max_examples=5, deadline=timedelta(seconds=4))
def test_blockify_1(maxnum: int):
    counter = count()
    cnext = lambda: next(counter)
    srcblk = blockify("source", cnext, in_type=None, out_type=np.int32)
    first_n_match(srcblk, np.int32, maxnum, list(range(maxnum)))
