from contextlib import redirect_stdout
from io import StringIO
import random
from typing import Optional, List, TypeVar, Union, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray
from typeguard import typechecked

from pcdr._internal.wavegen import noisify
from pcdr._internal.wavegen import TimeData


T = TypeVar("T")


@typechecked
def generate_ook_modulated_example_data(
    noise: bool = False, message_delay: bool = False, text_source: Optional[str] = None
) -> NDArray[np.complex64]:
    """
    Generate a file with the given `output_filename`.

    if `noise` is True, random noise will be added to the generated signal.
    if `message_delay` is True, there will be a pause before the meaningful data starts.
    if `text_source` is any string, a random sentence from it will be used as the message.

    Example usage:

    text_content = "These are some words, and more words. There are many words in a row in these sentences."
    generate_ook_modulated_example_data(text_source=text_content)
    """
    ## Avoid circular imports
    from pcdr._internal.wavegen import multiply_by_complex_wave

    message = "This is an example message."
    if text_source == None:
        print(
            f"No text source file specified, so all generated files will contain the message '{message}'"
        )
    else:
        assert isinstance(text_source, str)
        sentences = text_source.split(".")
        message = random.choice(sentences) + "."

    samp_rate = random.randrange(100, 700, 100)
    bit_length = random.randrange(50, 3000, 10)
    freq = random.randrange(10, samp_rate // 5)

    with_preamb = "«" + message
    bits = str_to_bin_list(with_preamb)
    baseband_sig = ook_modulate(bits, bit_length)
    fully_modded = multiply_by_complex_wave(baseband_sig, samp_rate, freq)

    if message_delay:
        fm2 = np.concatenate(
            [np.zeros(random.randint(100, 1500), dtype=np.complex64), fully_modded.y]
        )
    else:
        fm2 = fully_modded

    assert isinstance(fm2, np.ndarray)
    assert fm2.dtype == np.complex64

    if noise:
        fm3 = noisify(fm2)
    else:
        fm3 = fm2

    assert fm3.dtype == np.complex64
    return fm3


@typechecked
def generate_ook_modulated_example_file(
    output_filename: str,
    noise: bool = False,
    message_delay: bool = False,
    text_source: Optional[str] = None,
):
    """
    Generate a file with the given `output_filename`.

    if `noise` is True, random noise will be added to the generated signal.
    if `message_delay` is True, there will be a pause before the meaningful data starts.
    if `text_source` is any string, a random sentence from it will be used as the message.

    Example usage:

    text_content = "These are some words, and more words. There are many words in a row in these sentences."
    generate_ook_modulated_example_file("generated_example_file.complex", text_source=text_content)
    """

    data = generate_ook_modulated_example_data(noise, message_delay, text_source)
    data.tofile(output_filename)


@typechecked
def bytes_to_bin_list(b: Union[bytes, List[int]]) -> List[int]:
    """
    Converts each item in b to bits.

    Examples:

    >>> bytes_to_bin_list(b"C")
    [0, 1, 0, 0, 0, 0, 1, 1]

    >>> bytes_to_bin_list([67])
    [0, 1, 0, 0, 0, 0, 1, 1]

    >>> bytes_to_bin_list([192])
    [1, 1, 0, 0, 0, 0, 0, 0]

    >>> bytes_to_bin_list(b"CB")
    [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
    """
    assert all(0 <= item < 256 for item in b)
    bitstrs = [f"{c:08b}" for c in b]
    joined = "".join(bitstrs)
    result = list(map(int, joined))
    assert len(result) == len(b) * 8
    assert all(x in [0, 1] for x in result)
    return result


_NON_ASCII_ERR = "Currently, this only works for characters whose `ord` value is less than 256. For now, use `bytes_to_bin_list` if you wish to use non-ASCII characters. However, this may cause unexpected results for certain characters such as '«' that have multiple possible encodings."


@typechecked
def str_to_bin_list(message: str) -> List[int]:
    """
    Converts a string to a list of bits.

    Examples:

    >>> str_to_bin_list("C")
    [0, 1, 0, 0, 0, 0, 1, 1]

    >>> str_to_bin_list("CB")
    [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]

    >>> str_to_bin_list("«")
    [1, 0, 1, 0, 1, 0, 1, 1]
    """
    numeric = [ord(c) for c in message]
    if any(map(lambda x: x > 256, numeric)):
        raise ValueError(_NON_ASCII_ERR)
    return bytes_to_bin_list(numeric)


@typechecked
def int_to_bin_list(message: NDArray) -> List[int]:
    """
    Converts a numpy array of integers to a list of bits. Capable handling of a variety of dtypes.

    Examples:

    >>> int_to_bin_list(np.array([0x43], dtype=np.uint8))
    [0, 1, 0, 0, 0, 0, 1, 1]

    >>> int_to_bin_list(np.array([0x43, 0x42], dtype=np.uint8))
    [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]

    >>> int_to_bin_list(np.array([0x4342], dtype=np.uint16))
    [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
    """

    if message.dtype in [np.int8, np.uint8]:
        bitlength = 8
    elif message.dtype in [np.int16, np.uint16]:
        bitlength = 16
    elif message.dtype in [np.int32, np.uint32]:
        bitlength = 32
    elif message.dtype in [np.int64, np.uint64]:
        bitlength = 64
    else:
        raise ValueError("Unsupported dtype")

    ret = [0] * bitlength * len(message)
    bitlist_index = 0
    shift_list = list(reversed(range(0, bitlength)))
    for x in message:
        for bit_index in shift_list:
            if x & (1 << bit_index) > 0:
                ret[bitlist_index] = 1
            bitlist_index = bitlist_index + 1
    return ret


def __repeat_each_item(original: List[T], numtimes: int) -> List[T]:
    """
    Example:
    >>> __repeat_each_item([2, 5], 3)
    [2, 2, 2, 5, 5, 5]
    """
    assert 0 <= numtimes

    result = []
    for item in original:
        result += [item] * numtimes

    assert len(result) == len(original) * numtimes
    return result


def __must_be_binary(bits: List[int]) -> None:
    """
    This returns None:
    >>> __must_be_binary([1, 0, 1])

    This raises:
    >>> __must_be_binary("101")
    Traceback (most recent call last):
      ...
    pcdr.unstable.NonBitError: ...
    """
    from pcdr.unstable import NonBitError

    if not all(map(lambda x: x in [1, 0], bits)):
        raise NonBitError(
            "`bits` must be of type List[int], and all of those integers"
            ' must be either 0 or 1. It cannot be a string, such as "1010".'
        )


## Don't typecheck this one for now because we want to allow `bits`
## to have types that are other valid collections of int such as a numpy array of int
def ook_modulate(bits: List[int], bit_length: int, dtype=np.uint8) -> NDArray:
    """
    OOK Modulate. This is equivalent to simply repeating the bits, that is,
    >>> ook_modulate([1, 0], 3)
    array([1, 1, 1, 0, 0, 0], dtype=uint8)

    It also converts the data to a numpy array.

    Modulating it onto a carrier wave is unnecessary, as transmitting this
    to a GNU Radio osmocom sink will upconvert to the SDR peripheral's carrier frequency.

    If you want your OOK modulation to have a frequency at baseband before
    hardware upconversion, use the function `ook_modulate_at_frequency`.
    """
    __must_be_binary(bits)
    result = np.array(__repeat_each_item(bits, bit_length), dtype=dtype)
    assert isinstance(result, np.ndarray)
    assert result.dtype == dtype
    return result


## Don't typecheck this one for now; see note on `ook_modulate`
def ook_modulate_at_frequency(
    bits: List[int], bit_length: int, samp_rate: float, freq: float
) -> TimeData:
    """
    OOK Modulate at a given frequency. Returns the timestamps and the modulated data.

    Examples:
    >>> from pcdr._internal.basictermplot import plot
    >>> data = ook_modulate_at_frequency([1, 0, 1, 1], bit_length=20, samp_rate=40, freq=2)
    >>> plot(data.t, data.y.real, 80, 10)
    xmin: 0.00
    xmax: 1.98
    ymin: -1.00
    ymax: 1.00
    ~o███████████████████████████████████████o███████████████████o███████████████████
    ~█oo███████████████oo█████████████████████oo███████████████oo█oo███████████████oo
    ~███o█████████████o█████████████████████████o█████████████o█████o█████████████o██
    ~████████████████████████████████████████████████████████████████████████████████
    ~████o███████████o███████████████████████████o███████████o███████o███████████o███
    ~█████o█████████o████oooooooooooooooooooo█████o█████████o█████████o█████████o████
    ~██████o███████o███████████████████████████████o███████o███████████o███████o█████
    ~████████████████████████████████████████████████████████████████████████████████
    ~███████o█████o█████████████████████████████████o█████o█████████████o█████o██████
    ~████████ooooo███████████████████████████████████ooooo███████████████ooooo███████
    """
    __must_be_binary(bits)
    ## TODO: place this import better.
    ## For now, this import must be here due to circular imports
    from pcdr._internal.wavegen import multiply_by_complex_wave

    baseband_sig = ook_modulate(bits, bit_length)
    result = multiply_by_complex_wave(baseband_sig, samp_rate, freq)
    assert (
        result.t.dtype == np.float64
    )  # TODO: these may be unnecessary now that I know how to do NDArray type annotations properly
    assert result.y.dtype == np.complex64
    assert len(result.y) == len(result.y) == (len(bits) * bit_length)
    return result


def docstring_plot(xdata: list, ydata: list, xsize: int = 40, ysize: int = 7):
    """
    >>> import pytest; pytest.skip("Till github pcdr issue #36 is done")
    >>> import numpy as np
    >>> x = list(range(50))
    >>> print(docstring_plot(x, np.sin(x)))
         ┌─────────────────────────────────┐
     1.00┤▗▟  ▗▟   ▞▖  ▞▖  ▄▌  ▗▌  ▗▚   ▟  │
     0.33┤▌▝▖ ▌ ▌ ▞ ▌ ▐ ▌ ▐ ▚ ▗▘▐ ▗▘▐  ▞▝▖ │
    -0.33┤  ▌▗▘ ▚ ▌ ▚ ▌ ▝▖▞ ▐ ▐ ▝▖▐  ▌▐  ▚ │
    -1.00┤  ▝▟   ▜   ▚▘  ▝▌  ▚▌  ▝▞  ▝▞   ▚│
         └┬───────┬───────┬───────┬───────┬┘
         0.0    12.2    24.5    36.8   49.0
    >>> print(docstring_plot(x, np.sin(x), xsize=len(x), ysize=20))
         ┌───────────────────────────────────────────┐
     1.00┤ ▗    ▗▌   ▗▌    ▗     ▗    ▟     ▖        │
         │ █    ▐▌   ▐▚    ▛▖   ▗▜    █    ▐▌   ▗▀▌  │
         │▐▐    ▌▚   ▐▐    ▌▌   ▞▐    ▌▌   ▐▐   ▐ ▌  │
     0.67┤▐▐   ▗▘▐   ▐ ▌   ▌▌   ▌▝▖   ▌▌   ▞▐   ▐ ▌  │
         │▐ ▌  ▐ ▐   ▐ ▌  ▐ ▌   ▌ ▌  ▐ ▐   ▌▐   ▐ ▐  │
     0.33┤▌ ▌  ▐  ▌  ▞ ▌  ▐ ▐   ▌ ▌  ▐ ▐   ▌▐   ▌ ▐  │
         │▌ ▌  ▐  ▌  ▌ ▌  ▐ ▐  ▐  ▐  ▐ ▐  ▗▘ ▌  ▌ ▐  │
         │▌ ▐  ▌  ▌  ▌ ▐  ▞ ▐  ▐  ▐  ▐ ▐  ▐  ▌  ▌  ▌ │
    -0.00┤▘ ▐  ▌  ▌  ▌ ▐  ▌ ▝▖ ▐  ▐  ▌  ▌ ▐  ▌ ▗▘  ▌ │
         │  ▐  ▌  ▌ ▐  ▐  ▌  ▌ ▐  ▐  ▌  ▌ ▐  ▚ ▐   ▌ │
         │  ▐  ▌  ▌ ▐  ▝▖ ▌  ▌ ▌  ▐  ▌  ▌ ▌  ▐ ▐   ▌ │
    -0.33┤   ▌▐   ▌ ▐   ▌▐   ▌ ▌  ▐  ▌  ▌ ▌  ▐ ▐   ▌ │
         │   ▌▐   ▚ ▐   ▌▐   ▐ ▌  ▐ ▐   ▐ ▌  ▐ ▐   ▌ │
    -0.67┤   ▌▐   ▐ ▌   ▌▐   ▐ ▌  ▝▖▐   ▐ ▌   ▌▐   ▌ │
         │   ▚▐   ▝▖▌   ▌▐   ▐ ▌   ▌▞   ▐▐    ▌▐   ▚ │
         │   ▝▟    █    ▙▘    ▚▌   ▐▌   ▐▞    ▌▞   ▝▖│
    -1.00┤    ▝    ▜    ▝          ▝▌   ▝▌    ▝     ▝│
         └┬──────────┬─────────┬──────────┬─────────┬┘
         0.0       12.2      24.5       36.8     49.0
    """
    import plotext
    plotext.theme("clear")
    plotext.plot_size(xsize, ysize)
    plotext.plot(xdata, ydata)
    with redirect_stdout(StringIO()) as stream:
        plotext.show()
    return stream.getvalue().replace("\x1b[0m", "")[:-1]
