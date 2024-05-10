from gnuradio import gr, blocks
from typeguard import typechecked
from osmosdr import source as osmo_source
import numpy as np
from queue import Empty, SimpleQueue, LifoQueue
from typing import Optional, Union
from pcdr.v0_compat.helpers import validate_hack_rf_receive
import time




class SingleItemStack:
    def __init__(self):
        self.__stack = LifoQueue()
    
    def get(self, block: bool = True, timeout: Optional[float] = None):
        """Get the item stored in the stack.
        For details about arguments, see the Python Queue `.get()` docs:
        import queue
        q = queue.Queue()
        q.get()
        """
        return self.__stack.get(block, timeout)
    
    def put(self, val):
        """
        Put an item in the stack.
        If there's already an item, `get()` it to clear the stack, then proceed with `put()`.
        """
        try:
            self.__stack.get(block=False)
        except Empty:
            pass
        self.__stack.put(val)



class Blk_strength_at_freq(gr.sync_block):
    @typechecked
    def __init__(self, samp_rate: float, freq_of_interest: float, fft_size: int, avg_count: int = 1):
        gr.sync_block.__init__(self,
            name='Python Block: Strength at frequency',
            in_sig=[(np.complex64, fft_size)],
            out_sig=[]
        )
        assert 0 <= freq_of_interest < samp_rate / 2
        maxval = samp_rate/2 - samp_rate/fft_size
        ratio = fft_size / (2 * maxval)
        self._reading = SingleItemStack()
        self._fft = None
        self._idx = int(ratio * freq_of_interest)
        # self.__last_few = []
        # self._avg_count = avg_count
    
    def work(self, input_items, output_items):
        dat = input_items[0][0]
        self._fft = abs(np.fft.fft(dat))
        self._reading.put(float(self._fft[self._idx]))
        # if len(self.__last_few) < self._avg_count:
        #     fft_val = float(self._fft[self._idx])
        #     self.__last_few.append(fft_val)
        # else:
        #     avg = sum(self.__last_few) / len(self.__last_few)
        #     self._deq.append(avg)
        #     self.__last_few = []
        return 1



class OsmosdrReceiver:
    """A simplified interface to the Osmosdr Source
    which measures the strength of only the specified frequency.
    Example usage:
    
    ```python3
    import pcdr.simple
    receiver = pcdr.simple.OsmosdrReceiver("hackrf", 103.9e6)
    strength = receiver.get_strength()
    print(strength)
    ```
    """
    @typechecked
    def __init__(self, device_name: str, freq: float, *, device_id: Union[int, str] = 0):
        """
        `device_name`: One of the supported osmocom devices, such as "hackrf", "bladerf", etc. See the osmocom docs for a full list.
        `freq`: The frequency which the device will tune to. See note on `set_freq` for more info.
        `device_id`: A zero-based index ("0", "1", etc), or the partial serial number of the device, which can be gotten from GQRX
        >>> 
        """
        self.tb = gr.top_block()
        self.freq_offset = 20e3
        self.samp_rate = 2e6
        self.fft_size = 1024
        if_gain = 32
        bb_gain = 40
        device_args = f"{device_name}={device_id}"
        validate_hack_rf_receive(device_name, self.samp_rate, freq, if_gain, bb_gain)
        self.osmo_source = osmo_source(args=device_args)
        self.osmo_source.set_sample_rate(self.samp_rate)
        self.osmo_source.set_gain(0)
        self.osmo_source.set_if_gain(if_gain)
        self.osmo_source.set_bb_gain(bb_gain)
        self.stream_to_vec = blocks.stream_to_vector(gr.sizeof_gr_complex, self.fft_size)
        self.streng = Blk_strength_at_freq(self.samp_rate, self.freq_offset, self.fft_size, 10)
        self.tb.connect(self.osmo_source, self.stream_to_vec, self.streng)
        self.tb.start()
        self.set_freq(freq)


    @typechecked
    def get_strength(self, block: bool = True, timeout: float = 2.0) -> float:
        """Get the signal strength at the current frequency.
        The frequency is specified when the `OsmosdrReceiver` is created,
        and can be changed using `set_freq`.
        """
        return self.streng._reading.get(block, timeout)
    
    @typechecked
    def set_freq(self, freq: float, seconds: float = 0.5):
        """
        Set the frequency of the receiver, then
        wait `seconds` seconds. This delay allows the buffer to fill with data from the newly chosen frequency.
        
        Note: It's possible that a smaller delay would suffice; we chose a number that would safely guarantee a fresh buffer.
        
        Implementation detail: Those who are familiar with SDRs may wonder how
        this avoids the "DC spike". `set_freq` actually tunes below the specified
        frequency (20 kHz below at time of writing). Then, when `get_strength` is run,
        the receiver checks for activity at the expected location (the `freq` specified in this function).
        As a result, the strength level returned by `get_strength` is that of the desired frequency.
        """
        validate_hack_rf_receive("hackrf", center_freq=freq)
        # Also, TODO:
        #   Tell the queue work function to consume the entire input_items so that
        #   any data after that is fresh, THEN clear the current get_strength() reading (which
        #   will presumably be using a deque object).
        retval = self.osmo_source.set_center_freq(freq - self.freq_offset)
        time.sleep(seconds)
        return retval

    @typechecked
    def set_if_gain(self, if_gain: float):
        validate_hack_rf_receive("hackrf", if_gain=if_gain)
        self.osmo_source.set_if_gain(if_gain)

    @typechecked
    def set_bb_gain(self, bb_gain: float):
        validate_hack_rf_receive("hackrf", bb_gain=bb_gain)
        self.osmo_source.set_bb_gain(bb_gain)
