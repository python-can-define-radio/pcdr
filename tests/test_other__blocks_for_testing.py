from typing import List
from io import StringIO
from unittest.mock import patch
from gnuradio import gr, analog, blocks
import numpy as np
from typeguard import typechecked, TypeCheckError
from pcdr._internal.our_GR_blocks import Blk_sink_print
from pcdr._internal.basictermplot import plot
from pcdr._internal.misc import getSize
from pcdr.unstable.misc2 import first_n_match
import pytest
from itertools import count
from hypothesis import given, settings
from hypothesis.strategies import integers
from datetime import timedelta






class Blk_mult_three(gr.sync_block):
    @typechecked
    def __init__(self, in_sig: List[type], out_sig: List[type]):
        """in_sig used to be [np.uint8];
        out_sig used to be [np.float32]"""
        gr.sync_block.__init__(self, 'Multiply by three', in_sig, out_sig)

    def work(self, input_items, output_items):
        output_items[0][0] = 3 * input_items[0][0]
        return 1


class Blk_source_output_arb_num(gr.sync_block):
    """Outputs 0, 1, 2, 3, ... (as np.float32)"""
    def __init__(self):
        gr.sync_block.__init__(self, name="", in_sig=[], out_sig=[np.float32])
        self.i = 0

    def work(self, input_items, output_items):
        output_items[0][0] = self.i
        self.i += 1
        return 1


class Blk_arb_num_multiplied_by_three(gr.hier_block2):
    def __init__(self):
        """
        A simple hier block that outputs 0, 3, 6, ...; largely for a model for others.
        type: np.float32
        """
        gr.hier_block2.__init__(
            self, 
            "arb num, mult by three",
            gr.io_signature(0, 0, 0),
            gr.io_signature(1, 1, gr.sizeof_float)
        )

        self.arb = Blk_source_output_arb_num()
        self.mul3 = Blk_mult_three([np.float32], [np.float32])
        self.connect(self.arb, self.mul3, self)


class Blk_fake_osmosdr_source(gr.hier_block2):
    @typechecked
    def __init__(self, fake_samp_rate: float, fake_center_freq: float):
        """Return a block that resembles the osmosdr_source, but always has activity at only 200kHz."""
        
        gr.hier_block2.__init__(
            self, 
            "Fake osmocom source",
            gr.io_signature(0, 0, 0),
            gr.io_signature(1, 1, gr.sizeof_gr_complex)
        )

        self.fake_samp_rate = fake_samp_rate
        self.fake_center_freq = fake_center_freq
        fake_activity_freq = 200e3
        
        self.analog_sig_source = analog.sig_source_c(
            fake_samp_rate, analog.GR_COS_WAVE, 
            fake_activity_freq - fake_center_freq, 1)
        
        self.connect(self.analog_sig_source, self)

    @typechecked
    def set_sample_rate(self, samp_rate: float):
        assert self.fake_samp_rate == samp_rate

    @typechecked
    def set_center_freq(self, center_freq: float):
        assert self.fake_center_freq == center_freq
        
    @typechecked
    def set_gain(self, g: float):
        pass

    @typechecked
    def set_if_gain(self, g: float):
        pass

    @typechecked
    def set_bb_gain(self, g: float):
        pass


def test_first_n_match_1():
    with pytest.raises(TypeCheckError):
        first_n_match(
            "stuff",  # not a block
            np.float32,
            4,
            np.array([0, 1, 2, 3]))


def test_first_n_match_2():
    with pytest.raises(RuntimeError):
        first_n_match(
            Blk_source_output_arb_num(),
            np.complex64,  # wrong size of type (it does not check the actual type at time of writing)
            4,  
            [0, 1, 2, 3]  
        )


def test_first_n_match_3():
    with pytest.raises(AssertionError):
        first_n_match(
            Blk_source_output_arb_num(),
            np.float32,
            5,  # wrong length
            [0, 1, 2, 3]  
        )


def test_first_n_match_4():
    with pytest.raises(AssertionError):
        first_n_match(
            Blk_source_output_arb_num(),
            np.float32,
            4,
            [0, 1, 2, 5]  # wrong data
        )


@pytest.mark.slow
@given(integers(min_value=2, max_value=20))
@settings(max_examples=10, deadline=timedelta(milliseconds=500))
def test_Blk_source_output_arb_num_1(n: int):
    first_n_match(
        Blk_source_output_arb_num(),
        np.float32,
        n,
        list(range(n))
    )


@pytest.mark.slow
@given(integers(0, 10_000))
@settings(max_examples=3, deadline=timedelta(seconds=2))
def test_Blk_arb_num_multiplied_by_three(n: int):
    first_n_match(Blk_arb_num_multiplied_by_three(),
                  np.float32,
                  n,
                  list(range(0, n*3, 3)))



def test_Blk_fake_osmosdr_source():
    fake_center_freq = 170e3
    numSamples = 50  # arbitrary
    tb = gr.top_block()
    osm = Blk_fake_osmosdr_source(2e6, fake_center_freq)
    hea = blocks.head(getSize(np.complex64), numSamples)
    pri = Blk_sink_print(dtype=np.complex64)
    tb.connect(osm, hea, pri)
    output = StringIO()
    with patch('sys.stdout', output):
        tb.start()
        tb.wait()
    outv = output.getvalue()
    assert len(outv) > 0
    spl_nolast = outv.split("\n")[:-1]
    assert len(spl_nolast) == numSamples
    parsed = np.array(list(map(complex, spl_nolast)))
    plotOutput = StringIO()
    x = np.array(range(len(parsed)))
    y = parsed.real
    plot(x, y, output_stream=plotOutput)
    ## TODO: I haven't verified that the wave has the right frequency.
    assert plotOutput.getvalue() == """\
xmin: 0.00
xmax: 49.00
ymin: -1.00
ymax: 1.00
~o█████████████████████████████████████████████████
~█oooooooo█████████████████████████████████████████
~█████████ooo██████████████████████████████████████
~████████████oooo██████████████████████████████████
~████████████████ooo██████████████████████████████o
~███████████████████ooo████████████████████████ooo█
~██████████████████████oooo████████████████oooo████
~██████████████████████████oooooooooooooooo████████
"""
