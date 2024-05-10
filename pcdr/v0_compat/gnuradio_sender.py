import numpy as np
from gnuradio import gr
import sys
import signal
from typing import Tuple
from typeguard import typechecked
from gnuradio import blocks
import osmosdr



class DeviceParameterError(ValueError):
    pass


@typechecked
def validate_hack_rf_transmit(device_name: str,
                              samp_rate: float,
                              center_freq: float,
                              if_gain: int):
    if device_name != "hackrf":
        return
    
    if not (2e6 <= samp_rate <= 20e6):
        raise DeviceParameterError(
            "The HackRF One is only capable of sample rates "
            "between 2 Million samples per second (2e6) and "
            "20 Million samples per second (20e6). "
            f"Your specified sample rate, {samp_rate}, was outside of this range."
        )
    
    if not (1e6 < center_freq < 6e9):
        raise DeviceParameterError(
            "The HackRF One is only capable of center frequencies "
            "between 1 MHz (1e6) and 6 GHz (6e9). "
            f"Your specified frequency, {center_freq}, was outside of this range."
        )
    
    if not if_gain in range(0, 47+1, 1):
        raise DeviceParameterError(
            "The HackRF One, when in transmit mode, is only capable "
            "of the following if gain settings: "
            "[0, 1, 2, ... 45, 46, 47]. "
            f"Your specified if gain, {if_gain}, was not one of these options."
        )



def configure_graceful_exit(tb: gr.top_block):
    """The portion of GNU Radio boilerplate that 
    catches SIGINT and SIGTERM, and tells the flowgraph
    to gracefully stop before exiting the program.
    
    Used mainly for non-graphical flowgraphs."""
    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)


class vector_to_osmocom_sink(gr.top_block):

    def __init__(self,
                 data: Tuple[complex],
                 center_freq: float,
                 samp_rate: float,
                 if_gain: int,
                 device_args: str,
                 repeat: bool):
        """device_args: "hackrf=0" is common."""

        ## TODO: make the device name be passed in rather than always "hackrf"
        validate_hack_rf_transmit("hackrf", samp_rate, center_freq, if_gain)

        gr.top_block.__init__(self, "Top block")
        
        self.vector_source = blocks.vector_source_c(data, repeat)

        self.osmosdr_sink = osmosdr.sink(args=device_args)
        self.osmosdr_sink.set_sample_rate(samp_rate)
        self.osmosdr_sink.set_center_freq(center_freq)
        self.osmosdr_sink.set_gain(0)
        self.osmosdr_sink.set_if_gain(if_gain)
        self.osmosdr_sink.set_bb_gain(0)

        self.connect(self.vector_source, self.osmosdr_sink)


def gnuradio_send(data: np.ndarray,
                  center_freq: float,
                  samp_rate: float,
                  if_gain: int = 16,
                  device_args: str = "hackrf=0",
                  repeat: bool = False):
    """Sends `data` to osmocom sink."""
    normal_py_data = tuple(map(complex, data))  # GNURadio type issues. Eventually, fix this for efficiency
    tb = vector_to_osmocom_sink(normal_py_data, center_freq, samp_rate, if_gain, device_args, repeat)
    configure_graceful_exit(tb)
    tb.start()
    tb.wait()