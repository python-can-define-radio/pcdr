import numpy as np
from numpy.typing import NDArray
from gnuradio import gr
from gnuradio import blocks
import osmosdr
import time
from queue import Empty, SimpleQueue, LifoQueue
from pcdr._internal.misc import (
    SimpleQueueTypeWrapped, QueueTypeWrapped,
    queue_to_list, getSize
)
from typing import List, Optional
from typeguard import typechecked



class queue_source(gr.sync_block):
    ## TODO: Is this used anywhere? Let's start using Blk_queue_source instead

    def __init__(self, external_queue: SimpleQueueTypeWrapped, chunk_size: int, out_type = np.complex64):
        assert external_queue.qtype == np.ndarray
        assert external_queue.dtype == out_type
        assert external_queue.chunk_size == chunk_size
        gr.sync_block.__init__(
            self,
            name='Python Block: Queue Source',
            in_sig=[],
            out_sig=[(out_type, chunk_size)]
        )
        self.__queue = external_queue
        self.__chunk_size = chunk_size


    def work(self, input_items, output_items):
        try:
            output_items[0][0][:] = self.__queue.get_nowait()
            return 1
        except Empty:
            return -1  # Block is done
    
    
    def queue_put(self, data):
        assert len(data) == self.__chunk_size
        self.__queue.put(data)


class Blk_queue_source(gr.sync_block):
    @typechecked
    ## TODO: Look at my notes to self to see if I wanted to remove this timeout arg
    def __init__(self, dtype: type, chunk_size: int, timeout: Optional[float] = None, approx_queue_size_bytes: int = 30_000_000):
        gr.sync_block.__init__(self,
            name='Python Block: Queue Source',
            in_sig=[],
            out_sig=[(dtype, chunk_size)]
        )
        self.queue = QueueTypeWrapped(int(approx_queue_size_bytes/chunk_size))
        self.marked_done = False
        

    def work(self, input_items, output_items):
        try:
            output_items[0][0][:] = self.queue.get(timeout=10e-6)
            return 1
        except Empty:
            if self.marked_done:
                return -1
            else:
                return 0
            # print("Queue is empty, block will now report 'done' to GNU Radio flowgraph")
            


class Blk_sink_print(gr.sync_block):
    def __init__(self, sleep_seconds: float = 1e-6, only_print_1_in: int = 1, dtype: type = np.float32):
        gr.sync_block.__init__(self, name='Print sink', in_sig=[dtype], out_sig=[])
        self.only_print_1_in = only_print_1_in
        self.count = 0
        self.sleep_seconds = sleep_seconds

    def work(self, input_items, output_items):
        if self.count == 0:
            print(input_items[0][0])
            time.sleep(self.sleep_seconds)
        self.count = (self.count + 1) % self.only_print_1_in
        return 1


print_sink = Blk_sink_print  # Temporary alias while migrating to new name

class string_file_sink(gr.sync_block):

    def __init__(self, filename):
        gr.sync_block.__init__(
            self,
            name="Python Block: String File Sink",
            in_sig=[np.complex64],
            out_sig=[]
        )
        self.f = open(filename, "w")
        
    def work(self, input_items, output_items):
        singleDataPoint = input_items[0][0]

        self.f.write(f"{singleDataPoint}, ")
        self.f.flush()

        return 1


class queue_sink(gr.sync_block):
    ## TODO: Is this used anywhere? Let's start using Blk_queue_sink instead
    def __init__(self, chunk_size: int):
        gr.sync_block.__init__(
            self,
            name='Python Block: Data Queue Sink',
            in_sig=[(np.complex64, chunk_size)],
            out_sig=[]
        )
        self.__queue = SimpleQueueTypeWrapped(np.ndarray, np.complex64, chunk_size)
        self.__chunk_size = chunk_size


    def work(self, input_items, output_items):
        datacopy = input_items[0][0].copy()
        self.__queue.put(datacopy)
        return 1


    def get(self) -> np.ndarray:
        """Get a chunk from the queue of accumulated received data."""
        result = self.__queue.get()
        assert len(result) == self.__chunk_size
        return result

    def get_all(self) -> List[np.ndarray]:
        """Warning: this may or may not work while the flowgraph is running."""
        return queue_to_list(self.__queue)



class Blk_queue_sink(gr.sync_block):
    @typechecked
    def __init__(self, dtype: type, chunk_size: int):
        gr.sync_block.__init__(
            self,
            name='Python Block: Queue Sink',
            in_sig=[(dtype, chunk_size)],
            out_sig=[]
        )
        self.queue: SimpleQueue = SimpleQueue()

    def work(self, input_items, output_items):  
        datacopy = input_items[0][0].copy()
        self.queue.put(datacopy)
        return 1


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


class Averager:
    def __init__(self):
        raise NotImplementedError()



# class Blk_strength_at_freq(gr.sync_block):
#     @typechecked
#     def __init__(self, samp_rate: float, freq_of_interest: float, fft_size: int):
#         gr.sync_block.__init__(self,
#             name='Python Block: Strength at frequency',
#             in_sig=[(np.complex64, fft_size)],
#             out_sig=[]
#         )
#         assert 0 <= freq_of_interest < samp_rate / 2
#         maxval = samp_rate/2 - samp_rate/fft_size
#         ratio = fft_size / (2 * maxval)
#         self._reading = SingleItemStack()
#         self._fft = None
#         self._idx = int(ratio * freq_of_interest)
    
#     def work(self, input_items, output_items):
#         dat = input_items[0][0]
#         self._fft = abs(np.fft.fft(dat))
#         self._reading.put(float(self._fft[self._idx]))
#         return 1


class FreqComparisonMultiplier:
    @typechecked
    def __init__(self, samp_rate: float, freq_of_interest: float, chunk_size: int):
        from pcdr._internal.wavegen import make_wave  # Avoid circular imports
        wave = make_wave(samp_rate, freq_of_interest, "complex", num=chunk_size)
        self.__compare_wave: NDArray[np.complex64] = wave.y
        assert self.__compare_wave.dtype == np.complex64
    
    def get_freq_strength(self, dat: NDArray[np.complex64]) -> np.float32:
        """
        >>> from pcdr import make_wave
        >>> samp_rate = 100
        >>> freq_of_interest = 5
        >>> chunk_size = 1024
        >>> fcm = FreqComparisonMultiplier(samp_rate, freq_of_interest, chunk_size)
        >>> data1 = make_wave(samp_rate, 5, "complex", num=chunk_size)
        >>> fcm.get_freq_strength(data1)
        This needs to be filled
        >>> data2 = make_wave(samp_rate, 10, "complex", num=chunk_size)
        >>> fcm.get_freq_strength(data2)
        This needs to be filled
        """
        mul = dat * self.__compare_wave
        assert mul.dtype == np.complex64
        su = np.sum(mul)
        assert isinstance(su, np.complex64)  # TODO: remove for performance
        a = np.abs(su)
        assert isinstance(a, np.float32)  # TODO: remove for performance
        return a


class Blk_strength_by_mult(gr.sync_block):
    @typechecked
    def __init__(self, samp_rate: float, freq_of_interest: float, chunk_size: int):
        gr.sync_block.__init__(self,
            name='Python Block: Strength at frequency by mult',
            in_sig=[(np.complex64, chunk_size)],
            out_sig=[]
        )
        assert 0 <= freq_of_interest < samp_rate / 2
        self._reading = SingleItemStack()
        self._freq_comparison_multiplier = FreqComparisonMultiplier(samp_rate, freq_of_interest, chunk_size)
    
    def work(self, input_items, output_items):
        dat = input_items[0][0]
        streng = self._freq_comparison_multiplier.get_freq_strength(dat)
        self._reading.put(streng)
        return 1


class Blk_SingleItemStack(gr.sync_block):
    @typechecked
    def __init__(self, length: int, type_: type):
        gr.sync_block.__init__(self,
            name='Python Block: Single Item Stack',
            in_sig=[(type_, length)],
            out_sig=[]
        )
        self._reading = SingleItemStack()
    
    def work(self, input_items, output_items):
        dat = input_items[0][0]
        self._reading.put(dat)
        return 1


class Blk_VecSingleItemStack(gr.hier_block2):
    @typechecked
    def __init__(self, type_: type, vecsize: int):
                
        itemsize = getSize(type_)
        gr.hier_block2.__init__(
            self, 
            "VecSingleItemStack",
            gr.io_signature(1, 1, itemsize),
            gr.io_signature(0, 0, 0)
        )
        self.stv = blocks.stream_to_vector(itemsize, vecsize)
        self.sis = Blk_SingleItemStack(vecsize, type_)
        self.connect(self, self.stv, self.sis)