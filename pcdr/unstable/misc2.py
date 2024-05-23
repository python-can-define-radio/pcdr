"""Will eventually merge with misc.py

avoiding extra changes to that file for now."""


from itertools import count

from gnuradio import blocks, gr
import numpy as np
from typeguard import typechecked

from pcdr._internal.misc import getSize, connect_run_wait



class Blk_capture_as_list(gr.sync_block):
    @typechecked
    def __init__(self, type_: type, output: list):
        gr.sync_block.__init__(self, name="", in_sig=[type_], out_sig=[])
        self.captured: list = output
        self.type_ = type_ 

    def work(self, input_items, output_items):
        i00 = input_items[0][0]
        assert isinstance(i00, self.type_)
        self.captured.append(i00)
        return 1


@typechecked
def first_n_match(src_blk, output_type: type, n: int, compare: list):
    
    # This is actually a List[output_type], but mypy doesn't like that
    output: list = []  # type: ignore[var-annotated]
    connect_run_wait(
        src_blk,
        blocks.head(getSize(output_type), n),
        Blk_capture_as_list(output_type, output)
    )
    assert all(map(lambda x: isinstance(x, output_type), output))
    assert len(output) == len(compare)
    cmpnp = np.array(compare, dtype=output_type)  # type: ignore[var-annotated]
    for actual, expected, idx in zip(output, cmpnp, count()):
        assert isinstance(actual, output_type)
        assert isinstance(expected, output_type)
        assert actual == expected, f"Error at idx {idx}: Act {actual}, Exp {expected}"

