from typing import List
from io import StringIO
from unittest.mock import patch
from gnuradio import gr, analog, blocks
import numpy as np
from typeguard import typechecked
from pcdr._internal.our_GR_blocks import Blk_sink_print
from pcdr._internal.basictermplot import plot
from pcdr._internal.misc import getSize, connect_run_wait



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


class Blk_capture_as_list(gr.sync_block):
    @typechecked
    def __init__(self, type_: type, output: list):
        gr.sync_block.__init__(self, name="", in_sig=[type_], out_sig=[])
        self.captured = output

    def work(self, input_items, output_items):
        self.captured.append(input_items[0][0])
        return 1


def test_Blk_source_output_arb_num():
    numSamples = 4  # arbitrary
    output = []
    connect_run_wait(
        Blk_source_output_arb_num(),
        blocks.head(getSize(np.float32), numSamples),
        Blk_capture_as_list(np.float32, output)
    )
    eq: np.ndarray = (
          np.array(output) ==
          np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    assert eq.all()

    

@patch('sys.stdout', new_callable=StringIO)
def test_Blk_arb_num_multiplied_by_three(output: StringIO):
    numSamples = 4  # arbitrary
    tb = gr.top_block()
    arb3 = Blk_arb_num_multiplied_by_three()
    hea = blocks.head(getSize(np.float32), numSamples)
    pri = Blk_sink_print()
    tb.connect(arb3, hea, pri)
    tb.start()
    tb.wait()
    outv = output.getvalue()
    assert len(outv) > 0
    spl_nolast = outv.split("\n")[:-1]
    assert len(spl_nolast) == numSamples
    assert spl_nolast[:4] == [
        '0.0', '3.0', '6.0', '9.0'
    ]



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
