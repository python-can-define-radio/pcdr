from typing import Union, TypeVar, Protocol, Callable
import typing


## As per https://stackoverflow.com/questions/43957034/specifying-a-type-to-be-a-list-of-numbers-ints-and-or-floats
TRealNum = TypeVar("TRealNum", int, float)

TRealOrComplexNum = TypeVar("TRealOrComplexNum", int, float, complex)


# TODO: incorporate this # @typing.runtime_checkable
class HasWorkFunc(Protocol):
    work: Callable


## avoid circular imports, and avoid gnuradio dependency
## when its unneeded
try:
    from gnuradio import gr
    from PyQt5 import Qt

    class top_block_and_widget(gr.top_block, Qt.QWidget):
        pass

    class SupportsQueueSink(Protocol):
        queue_sink: "queue_sink"

    from pcdr._internal.our_GR_blocks import queue_sink

except ModuleNotFoundError:
    pass
