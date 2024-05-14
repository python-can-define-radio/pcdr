import random
from typing import Optional, List, Tuple, Literal
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from typeguard import typechecked

from pcdr._internal.fileio import writeRealCSV, writeComplexCSV
from pcdr._internal.modulators import ook_modulate
from pcdr._internal.helpers import str_to_bin_list



@dataclass
class TimeData:
    """Measurements and their associated timestamps."""
    t: NDArray[np.float64]
    """The timestamps that correspond to the `y` values."""
    y: NDArray
    """The 'y-values', i.e., the actual data which correspond to the times indicated by `t`."""
    @property
    def x(self) -> NDArray[np.float64]:
        """An alias for t (the timestamps)."""
        return self.t

    @x.setter
    def x(self, value: NDArray[np.float64]):
        """An alias for t (the timestamps)."""
        self.t = value
    

@typechecked
def make_timestamps_seconds(seconds: float, num_samples: int, dtype=np.float64) -> np.ndarray:
    """Creates timestamps from zero up to the given maximum number of seconds.
    Implemented using np.linspace().
    
    Note: We use np.float64 as the default dtype because np.float32 was causing float rounding issues
    that became worse with larger time values (as float rounding issues usually do).
    
    Example:
    >>> make_timestamps_seconds(2, 10)
    array([0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8])
    """
    assert 0 <= seconds
    assert 0 <= num_samples
    result = np.linspace(
        start=0,
        stop=seconds,
        num=num_samples,
        endpoint=False,
        dtype=dtype
    )
    assert result.dtype == dtype
    assert len(result) == num_samples
    assert (0 <= result).all()
    return result


@typechecked
def make_timestamps_samprate(samp_rate: float, num_samples: int, dtype=np.float64) -> NDArray:
    """Creates `num_samples` timestamps spaced by `1/samp_rate`.
    Implemented using np.linspace().
    
    Examples:
    >>> make_timestamps_samprate(5, 10)
    array([0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8])

    >>> make_timestamps_samprate(10, 4)
    array([0. , 0.1, 0.2, 0.3])

    Note: We use np.float64 as the default dtype because np.float32 was causing float rounding issues
    that became worse with larger time values (as float rounding issues usually do)."""
    assert 0 < samp_rate
    assert 0 <= num_samples
    
    result = np.linspace(
            start=0,
            stop=num_samples/samp_rate,
            num=num_samples,
            endpoint=False,
            dtype=dtype
        )
    
    assert result.dtype == dtype
    assert len(result) == num_samples
    assert (0 <= result).all()

    return result


@typechecked
def make_timestamps():
    raise NotImplementedError("Intended purpose: run either make_timestamps_seconds or make_timestamps_samprate based on provided arguments")

#### This is something we may do eventually.
# @overload
# def makeRealWave(samp_rate: float,
#                  freq: float,
#                  num_samples: int,
#                  allowAliasing: bool = False) -> Tuple[NDArray, NDArray]: ...
# @overload
# def makeRealWave(samp_rate: float,
#                  freq: float,
#                  seconds: float,
#                  allowAliasing: bool = False) -> Tuple[NDArray, NDArray]: ...
# def makeRealWave(samp_rate: float,
#                  freq: float,
#                  **kwargs):
#     return something


@typechecked
def makeRealWave_basic(timestamps: NDArray, freq: float) -> NDArray[np.float32]:
    """Return a sine wave.
    
    Example:
    >>> from pcdr._internal.basictermplot import plot
    >>> timestamps = make_timestamps_seconds(1, 50)
    >>> wave = makeRealWave_basic(timestamps, 2)
    >>> plot(timestamps, wave)
    xmin: 0.00
    xmax: 0.98
    ymin: -1.00
    ymax: 1.00
    ~██████o████████████████████████o██████████████████
    ~████oo█ooo██████████████████o█o█ooo███████████████
    ~██oo██████o████████████████oo██████o██████████████
    ~█o█████████o██████████████o█████████o█████████████
    ~o███████████oo███████████o███████████oo███████████
    ~██████████████o█████████o██████████████o█████████o
    ~███████████████o██████oo████████████████o██████oo█
    ~████████████████oooooo███████████████████oooooo███
    """
    result = np.sin(freq * 2 * np.pi * timestamps, dtype=np.float32)
    assert timestamps.shape == result.shape
    assert result.dtype == np.float32
    return result


@typechecked
def makeComplexWave_basic(timestamps: NDArray, freq: float) -> NDArray[np.complex64]:
    """Return a complex wave.
    
    The real part is cosine (starts at 1); the imaginary part is sine (starts at 0).
    
    Example:
    >>> from pcdr._internal.basictermplot import plot
    >>> timestamps = make_timestamps_seconds(1, 50)
    >>> wave = makeComplexWave_basic(timestamps, 2)
    >>> plot(timestamps, wave.real)
    xmin: 0.00
    xmax: 0.98
    ymin: -0.99
    ymax: 1.00
    ~o████████████████████████o████████████████████████
    ~█ooo██████████████████ooo█ooo██████████████████ooo
    ~████o████████████████o██████o█████████████████o███
    ~█████o██████████████o█████████o██████████████o████
    ~██████o████████████o███████████o████████████o█████
    ~███████o██████████o█████████████o██████████o██████
    ~████████oo██████oo███████████████oo██████oo███████
    ~██████████oooooo███████████████████oooooo█████████
    >>> plot(timestamps, wave.imag)
    xmin: 0.00
    xmax: 0.98
    ymin: -1.00
    ymax: 1.00
    ~██████o████████████████████████o██████████████████
    ~████oo█ooo██████████████████o█o█ooo███████████████
    ~██oo██████o████████████████oo██████o██████████████
    ~█o█████████o██████████████o█████████o█████████████
    ~o███████████oo███████████o███████████oo███████████
    ~██████████████o█████████o██████████████o█████████o
    ~███████████████o██████oo████████████████o██████oo█
    ~████████████████oooooo███████████████████oooooo███
    """
    ## Note: I don't know enough about math with complex numbers
    ## to know if freq should be restricted to real, but I figured
    ## it was better to type-annotate it as `float` rather than leaving
    ## it as `Any`.
    result = np.exp(1j * freq * 2 * np.pi * timestamps, dtype=np.complex64)
    assert timestamps.shape == result.shape
    assert result.dtype == np.complex64
    return result



class AliasError(ValueError):
    pass


@typechecked
def _isAliasingWhenDisallowed(allowAliasing: bool, freq: float, samp_rate: float) -> bool:
    """
    Examples:

    >>> allowAliasing = False
    >>> samp_rate = 5
    >>> too_high_freq = 4
    >>> acceptable_freq = 2
    >>> _isAliasingWhenDisallowed(allowAliasing, too_high_freq, samp_rate)
    True
    >>> _isAliasingWhenDisallowed(allowAliasing, acceptable_freq, samp_rate)
    False
    >>> allowAliasing = True
    >>> _isAliasingWhenDisallowed(allowAliasing, too_high_freq, samp_rate)
    False
    >>> _isAliasingWhenDisallowed(allowAliasing, acceptable_freq, samp_rate)
    False
    """
    return (not allowAliasing) and (abs(freq) > samp_rate/2)


@typechecked
def _aliasingError(allowAliasing: bool, freq: float, samp_rate: float) -> None:
    """Gives a detailed Aliasing error message if it's aliasing when it shouldn't.
    :raises AliasError:"""
    if _isAliasingWhenDisallowed(allowAliasing, freq, samp_rate):
        raise AliasError(f"For a sample rate of {samp_rate}, the highest frequency that can be faithfully represented is {samp_rate/2}. The specified freq, {freq}, is greater than the limit specified by Shannon/Nyquist/Kotelnikov/Whittaker (commonly called the Nyquist frequency).")


@typechecked
def makeComplexWave_numsamps(num_samples: int, samp_rate: float, freq: float, allowAliasing: bool = False) -> TimeData:
    """
    Returns a tuple (timestamps, wave).
    
    The real part of the wave is cosine (starts at 1); the imaginary part is sine (starts at 0).

    :raises AliasError: if _isAliasingWhenDisallowed
    
    Example:
    >>> from pcdr._internal.basictermplot import plot
    >>> timestamps, wave = makeComplexWave_numsamps(50, 50, 2)
    >>> plot(timestamps, wave.real)
    xmin: 0.00
    xmax: 0.98
    ymin: -0.99
    ymax: 1.00
    ~o████████████████████████o████████████████████████
    ~█ooo██████████████████ooo█ooo██████████████████ooo
    ~████o████████████████o██████o█████████████████o███
    ~█████o██████████████o█████████o██████████████o████
    ~██████o████████████o███████████o████████████o█████
    ~███████o██████████o█████████████o██████████o██████
    ~████████oo██████oo███████████████oo██████oo███████
    ~██████████oooooo███████████████████oooooo█████████
    >>> plot(timestamps, wave.imag)
    xmin: 0.00
    xmax: 0.98
    ymin: -1.00
    ymax: 1.00
    ~██████o████████████████████████o██████████████████
    ~████oo█ooo██████████████████o█o█ooo███████████████
    ~██oo██████o████████████████oo██████o██████████████
    ~█o█████████o██████████████o█████████o█████████████
    ~o███████████oo███████████o███████████oo███████████
    ~██████████████o█████████o██████████████o█████████o
    ~███████████████o██████oo████████████████o██████oo█
    ~████████████████oooooo███████████████████oooooo███
    """
    assert 0 < samp_rate
    assert 0 <= num_samples
    _aliasingError(allowAliasing, freq, samp_rate)
    t = num_samples / samp_rate
    timestamps = make_timestamps_seconds(seconds=t, num_samples=num_samples)
    wave = makeComplexWave_basic(timestamps, freq)
    assert len(timestamps) == len(wave) == num_samples
    return TimeData(timestamps, wave)


@typechecked
def makeRealWave_numsamps(num_samples: int, samp_rate: float, freq: float, allowAliasing: bool = False) -> TimeData:
    """
    Return a Real wave.

    :raises AliasError: if _isAliasingWhenDisallowed
    
    Example:
    >>> from pcdr._internal.basictermplot import plot
    >>> timestamps, wave = makeRealWave_numsamps(50, 50, 2)
    >>> plot(timestamps, wave)
    xmin: 0.00
    xmax: 0.98
    ymin: -1.00
    ymax: 1.00
    ~██████o████████████████████████o██████████████████
    ~████oo█ooo██████████████████o█o█ooo███████████████
    ~██oo██████o████████████████oo██████o██████████████
    ~█o█████████o██████████████o█████████o█████████████
    ~o███████████oo███████████o███████████oo███████████
    ~██████████████o█████████o██████████████o█████████o
    ~███████████████o██████oo████████████████o██████oo█
    ~████████████████oooooo███████████████████oooooo███
    """
    assert 0 < samp_rate
    assert 0 <= num_samples
    _aliasingError(allowAliasing, freq, samp_rate)
    t = num_samples / samp_rate
    timestamps = make_timestamps_seconds(seconds=t, num_samples=num_samples)
    wave = makeRealWave_basic(timestamps, freq)
    assert len(timestamps) == len(wave) == num_samples
    return TimeData(timestamps, wave)


def makeComplexWave_time(seconds: float, samp_rate: float, freq: float, allowAliasing: bool = False) -> TimeData:
    """
    Returns a tuple (timestamps, wave).
    
    The real part of the wave is cosine (starts at 1); the imaginary part is sine (starts at 0).

    :raises AliasError: if _isAliasingWhenDisallowed
    
    Example:
    >>> from pcdr._internal.basictermplot import plot
    >>> timestamps, wave = makeComplexWave_time(1, 50, 2)
    >>> plot(timestamps, wave.real)
    xmin: 0.00
    xmax: 0.98
    ymin: -0.99
    ymax: 1.00
    ~o████████████████████████o████████████████████████
    ~█ooo██████████████████ooo█ooo██████████████████ooo
    ~████o████████████████o██████o█████████████████o███
    ~█████o██████████████o█████████o██████████████o████
    ~██████o████████████o███████████o████████████o█████
    ~███████o██████████o█████████████o██████████o██████
    ~████████oo██████oo███████████████oo██████oo███████
    ~██████████oooooo███████████████████oooooo█████████
    >>> plot(timestamps, wave.imag)
    xmin: 0.00
    xmax: 0.98
    ymin: -1.00
    ymax: 1.00
    ~██████o████████████████████████o██████████████████
    ~████oo█ooo██████████████████o█o█ooo███████████████
    ~██oo██████o████████████████oo██████o██████████████
    ~█o█████████o██████████████o█████████o█████████████
    ~o███████████oo███████████o███████████oo███████████
    ~██████████████o█████████o██████████████o█████████o
    ~███████████████o██████oo████████████████o██████oo█
    ~████████████████oooooo███████████████████oooooo███
    """
    assert 0 < samp_rate
    _aliasingError(allowAliasing, freq, samp_rate)
    num_samples = int(samp_rate * seconds)
    timestamps = make_timestamps_seconds(seconds, num_samples)
    wave = makeComplexWave_basic(timestamps, freq)
    assert len(timestamps) == len(wave) == num_samples
    return TimeData(timestamps, wave)


def makeRealWave_time(seconds: float, samp_rate: float, freq: float, allowAliasing: bool = False) -> TimeData:
    """
    Return a Real wave.

    :raises AliasError: if _isAliasingWhenDisallowed
    
    Example:
    >>> from pcdr._internal.basictermplot import plot
    >>> timestamps, wave = makeRealWave_time(1, 50, 2)
    >>> plot(timestamps, wave.real)
    xmin: 0.00
    xmax: 0.98
    ymin: -1.00
    ymax: 1.00
    ~██████o████████████████████████o██████████████████
    ~████oo█ooo██████████████████o█o█ooo███████████████
    ~██oo██████o████████████████oo██████o██████████████
    ~█o█████████o██████████████o█████████o█████████████
    ~o███████████oo███████████o███████████oo███████████
    ~██████████████o█████████o██████████████o█████████o
    ~███████████████o██████oo████████████████o██████oo█
    ~████████████████oooooo███████████████████oooooo███
    """
    assert 0 < samp_rate
    _aliasingError(allowAliasing, freq, samp_rate)
    num_samples = int(samp_rate * seconds)
    timestamps = make_timestamps_seconds(seconds, num_samples)
    wave = makeRealWave_basic(timestamps, freq)
    assert len(timestamps) == len(wave) == num_samples
    return TimeData(timestamps, wave)


@typechecked
def make_wave(samp_rate: float,
             freq: float,
             type_: Literal["real", "complex"],
             *, seconds: Optional[float] = None,
             num: Optional[int] = None,
             allowAliasing: bool = False) -> TimeData:
    """
    Generate a sine wave and the associated timestamps.

    Parameters:
    -----------
    samp_rate :
        Measured in samples per second.
    freq :
        Frequency in Hz of the generated wave.
    type_ :
        "real" will have no imaginary part. "complex" will have an imaginary part that is 90 degrees out of phase from the real part.
    seconds : 
        The length of the signal will be this many seconds. See example 1.
        

    Examples:
    ---------

    Example 1: Demonstrating `seconds`.
    
    If we set our sample rate...   
    >>> samp_rate = 50

    ...and our number of seconds...  
    >>> seconds = 2

    ...the return value will have the length that we expect, which is samp_rate * seconds samples:
    >>> timestamps, wave = makeWave(samp_rate, freq=3, "real", seconds=seconds)
    >>> print(len(timestamps))
    100
    >>> print(len(wave))
    100
    

    Example 2: Plotting a result.
    Generate one second of data of a 3 Hz wave. The sample rate is 50 samples per second, and it is a real wave (no imaginary part).
    >>> timestamps, wave = makeWave(50, 3, "real", seconds=1)

    Plot that wave. (Note, matplotlib would generate a superior plot; this is a simplified textual plot.)
    >>> from pcdr._internal.basictermplot import plot
    >>> plot(timestamps, wave)
    xmin: 0.00
    xmax: 0.98
    ymin: -1.00
    ymax: 1.00
    ~████o████████████████o████████████████████████████
    ~███o█oo████████████oo█o█████████████oooo██████████
    ~██o████o██████████o████o███████████o████o█████████
    ~█o██████████████████████o█████████o██████o████████
    ~o███████o████████o███████o███████o████████o███████
    ~█████████o██████o█████████o██████████████████████o
    ~██████████o████o███████████o████o██████████o████o█
    ~███████████oooo█████████████o█oo████████████oooo██

    >>> timestamps, wave = makeWave(50, 3, "complex", seconds=1)
    >>> plot(timestamps, wave.real)
    xmin: 0.00
    xmax: 0.98
    ymin: -1.00
    ymax: 1.00
    ~o█████████████████████████████████████████████████
    ~█oo████████████oooo█████████████oooo████████████oo
    ~██████████████o████o███████████o████o█████████████
    ~███o█████████o██████o█████████o██████o█████████o██
    ~████o████████████████o██████o█████████████████o███
    ~█████o██████o█████████o█████o█████████o██████o████
    ~██████o████o███████████████████████████o████o█████
    ~███████oooo████████████ooooo████████████oooo██████

    >>> plot(timestamps, wave.imag)
    xmin: 0.00
    xmax: 0.98
    ymin: -1.00
    ymax: 1.00
    ~████o████████████████o████████████████████████████
    ~███o█oo████████████oo█o█████████████oooo██████████
    ~██o████o██████████o████o███████████o████o█████████
    ~█o██████████████████████o█████████o██████o████████
    ~o███████o████████o███████o███████o████████o███████
    ~█████████o██████o█████████o██████████████████████o
    ~██████████o████o███████████o████o██████████o████o█
    ~███████████oooo█████████████o█oo████████████oooo██

    >>> timestamps, wave = makeWave(50, 3, "real", num=60)
    >>> plot(timestamps, wave)
    xmin: 0.00
    xmax: 1.18
    ymin: -1.00
    ymax: 1.00
    ~████o████████████████o████████████████████████████████o█████
    ~███o█oo████████████oo█o█████████████oooo█████████████o█oo███
    ~██o████o██████████o████o███████████o████o███████████o████o██
    ~█o██████████████████████o█████████o██████o█████████o████████
    ~o███████o████████o███████o███████o████████o███████o██████o██
    ~█████████o██████o█████████o██████████████████████o█████████o
    ~██████████o████o███████████o████o██████████o████o███████████
    ~███████████oooo█████████████o█oo████████████oooo████████████

    >>> timestamps, wave = makeWave(50, 3, "complex", num=60)
    >>> plot(timestamps, wave.real)
    xmin: 0.00
    xmax: 1.18
    ymin: -1.00
    ymax: 1.00
    ~o█████████████████████████████████████████████████o█████████
    ~█oo████████████oooo█████████████oooo████████████oo█oo███████
    ~██████████████o████o███████████o████o███████████████████████
    ~███o█████████o██████o█████████o██████o█████████o█████o██████
    ~████o████████████████o██████o█████████████████o███████o█████
    ~█████o██████o█████████o█████o█████████o██████o█████████o████
    ~██████o████o███████████████████████████o████o███████████o███
    ~███████oooo████████████ooooo████████████oooo█████████████o█o
    >>> plot(timestamps, wave.imag)
    xmin: 0.00
    xmax: 1.18
    ymin: -1.00
    ymax: 1.00
    ~████o████████████████o████████████████████████████████o█████
    ~███o█oo████████████oo█o█████████████oooo█████████████o█oo███
    ~██o████o██████████o████o███████████o████o███████████o████o██
    ~█o██████████████████████o█████████o██████o█████████o████████
    ~o███████o████████o███████o███████o████████o███████o██████o██
    ~█████████o██████o█████████o██████████████████████o█████████o
    ~██████████o████o███████████o████o██████████o████o███████████
    ~███████████oooo█████████████o█oo████████████oooo████████████
    
    >> makeWave(10, 2, "real")
    Traceback:
      ...
    ValueError: Must specify either `seconds` or `num`
    >> makeWave(10, 2, "complex", seconds=3, num=60)
    Traceback:
      ...
    ValueError: Cannot specify both `seconds` or `num` simultaneously
    >> makeWave(10, 7, "real", seconds=3)
    Traceback:
      ...
    For a sample rate of 10, the highest frequency ...
    >> makeWave(10, 7, "real", seconds=3, allowAliasing=True) == makeRealWave_time(3, 10, 2, allowAliasing=True)
    True
    """
    if seconds != None and num != None:
        raise ValueError("Cannot specify both `seconds` and `num` simultaneously")
    elif seconds == None and num == None:
        raise ValueError("Must specify either `seconds` or `num`")
    elif seconds != None:
        assert isinstance(seconds, float)
        if type_ == "real":
            return makeRealWave_time(seconds, samp_rate, freq, allowAliasing)
        elif type_ == "complex":
            return makeComplexWave_time(seconds, samp_rate, freq, allowAliasing)
        else:
            raise ValueError("This will never happen if the @typechecked works")
    elif num != None:
        assert isinstance(num, int)
        if type_ == "real":
            return makeRealWave_numsamps(num, samp_rate, freq, allowAliasing)
        elif type_ == "complex":
            return makeComplexWave_numsamps(num, samp_rate, freq, allowAliasing)
        else:
            raise ValueError("This will never happen if the @typechecked works")
    else:
        raise Exception("Impossible case")


def wave_and_write(basename: str, timestamps: np.ndarray, freq, complex_or_real: Literal["c", "r"]):
    if complex_or_real == "r":
        data: NDArray[np.float32] = makeRealWave_basic(timestamps, freq)
        writeRealCSV(basename + ".csv", data)
        data.tofile(basename + ".float32")
    elif complex_or_real == "c":
        data: NDArray[np.complex64] = makeComplexWave_basic(timestamps, freq)
        writeComplexCSV(basename + ".csv", data)
        data.tofile(basename + ".complex64")
    else:
        raise ValueError("Must choose 'c' or 'r' to specify if real or complex is wanted.")


def wave_file_gen_prompts():
    print()
    print("This will create a simulated wave, and write it to two files:")
    print(" - A CSV file (for easy viewing in text editors and spreadsheet programs)")
    print(" - Either a raw float32 or complex64 file (for use in GNU Radio, URH, etc)")
    print()

    samp_rate = float(input("Pick a sample rate (samples per second): "))
    max_time = float(input("How many seconds of data would you like to generate? "))
    num_samples_original = samp_rate * max_time
    num_samples = int(num_samples_original)

    if num_samples != num_samples_original:
        raise ValueError(f"The number of samples would be {num_samples_original}, but a partial sample is meaningless.\nPlease pick a sample rate and an amount of time whose product is an integer.")

    freq = float(input("What frequency wave would you like to generate (Hz)? "))
    complex_or_real = input("Complex or Real wave? Enter c or r. ")
    filename = input("Filename? (Press enter to choose the default name, 'generated_data'.) ")
    if filename.strip() == "":
        filename = "generated_data"

    timestamps = make_timestamps_seconds(max_time, num_samples)
    print("------------------")
    print(f"Going to generate {int(num_samples)} samples.")
    print("Simulated samples were taken at these times (units are seconds):")
    print(timestamps)

    wave_and_write(filename, timestamps, freq, complex_or_real)
    print("Done writing files.")


def wave_file_gen(samp_rate: float, max_time: float, freq: float, complex_or_real: Literal["c", "r"], filename: str = 'generated_data'):
    """Units:
    samp_rate: samples per sec
    max_time: seconds
    freq: Hz
    complex_or_real: 'c' or 'r'
    """
    
    num_samples_f = samp_rate * max_time
    num_samples = int(num_samples_f)

    if num_samples != num_samples_f:
        raise ValueError(f"The number of samples would be {num_samples}, but a partial sample is meaningless.\nPlease pick a sample rate and an amount of time whose product is an integer.")

    timestamps = make_timestamps_seconds(max_time, num_samples)

    wave_and_write(filename, timestamps, freq, complex_or_real)


@typechecked
def multiply_by_complex_wave(baseband_sig: NDArray, samp_rate: float, freq: float, allowAliasing: bool = False) -> TimeData:
    """
    Returns a tuple (timestamps, mult).

    >>> from pcdr._internal.basictermplot import plot
    >>> from pcdr import multiply_by_complex_wave, ook_modulate
    >>> baseband_sig = ook_modulate([1, 0], 32)
    >>> timestamps, mult = multiply_by_complex_wave(baseband_sig, 64, 2)
    >>> plot(timestamps, mult.real)
    xmin: 0.00
    xmax: 0.98
    ymin: -1.00
    ymax: 1.00
    ~o███████████████████████████████████████████████████████████████
    ~█ooo█████████████████████████ooo████████████████████████████████
    ~████oo█████████████████████oo███████████████████████████████████
    ~██████oo█████████████████oo█████████████████████████████████████
    ~████████o███████████████o███████oooooooooooooooooooooooooooooooo
    ~█████████oo███████████oo████████████████████████████████████████
    ~███████████oo███████oo██████████████████████████████████████████
    ~█████████████ooooooo████████████████████████████████████████████
    """
    wave = makeComplexWave_numsamps(len(baseband_sig), samp_rate, freq, allowAliasing)
    mult = np.complex64(baseband_sig) * wave.y
    assert len(wave.t) == len(mult) == len(wave.y)
    assert wave.t.dtype == np.float64
    assert mult.dtype == np.complex64
    return TimeData(wave.t, mult)


@typechecked
def multiply_by_real_wave(baseband_sig: NDArray, samp_rate: float, freq: float, allowAliasing: bool = False) -> TimeData:
    """
    Returns a tuple (timestamps, mult).

    >>> from pcdr._internal.basictermplot import plot
    >>> from pcdr import multiply_by_real_wave, ook_modulate
    >>> baseband_sig = ook_modulate([1, 0], 32)
    >>> timestamps, wave = multiply_by_real_wave(baseband_sig, 64, 2)
    >>> plot(timestamps, wave)
    xmin: 0.00
    xmax: 0.98
    ymin: -1.00
    ymax: 1.00
    ~████████o███████████████████████████████████████████████████████
    ~█████ooo█ooo████████████████████████████████████████████████████
    ~███oo███████oo██████████████████████████████████████████████████
    ~█oo███████████oo████████████████████████████████████████████████
    ~o███████████████o███████████████oooooooooooooooooooooooooooooooo
    ~█████████████████oo███████████oo████████████████████████████████
    ~███████████████████oo███████oo██████████████████████████████████
    ~█████████████████████ooooooo████████████████████████████████████
    """
    wave = makeRealWave_numsamps(len(baseband_sig), samp_rate, freq, allowAliasing)
    mult = np.float32(baseband_sig) * wave.y
    assert len(wave.t) == len(mult) == len(wave.y)
    assert wave.t.dtype == np.float64
    assert mult.dtype == np.float32
    return TimeData(wave.t, mult)


@typechecked
def random_normal(size: int, dtype=np.float32, seed=None) -> NDArray:
    """A wrapper of numpy's `standard_normal()` function;
    returns a numpy array of length `size` containing normally distributed noise.

    `seed` is optional, and mostly just used for testing the function.
    
    TODO >>> random_normal(size=3, seed=0)
    array([ 1.117622 , -1.3871249, -0.4265716], dtype=float32)

    TODO >>> random_normal(size=2, dtype=np.float64, seed=0)
    array([ 0.12573022, -0.13210486])
    """
    rng = np.random.default_rng(seed=seed)
    re = rng.standard_normal(size=size, dtype=np.float32)
    if dtype == np.float32:
        result = rng.standard_normal(size=size, dtype=np.float32)
    elif dtype == np.complex64:
        im = 1j * rng.standard_normal(size=size, dtype=np.float32)
        result = re + im  # type: ignore[assignment]
    else:
        raise NotImplementedError()
    assert isinstance(result, np.ndarray)
    assert len(result) == size
    assert result.dtype == dtype
    return result


@typechecked
def noisify(data: NDArray, amplitude=1, seed=None) -> NDArray:
    """
    Returns a copy of `data` with random normally distributed noise added.
    `seed` is optional, and mostly just used for testing the function.

    TODO >>> dat = np.array([10, 100, 1000], dtype=np.float32)
    TODO >>> noisify(dat, amplitude=0.1, seed=0)
    array([ 11.117622,  98.61288 , 999.5734  ], dtype=float32)

    TODO >>> dat = np.array([10 + 20j, 100 + 200j], dtype=np.complex64)
    TODO >>> noisify(dat, amplitude=0.1, seed=0)
    array([11.117622 +21.117622j, 98.61288 +198.61287j ], dtype=complex64)
    """
    if data.dtype == np.float32:
        randnoise = random_normal(len(data), dtype=np.float32, seed=seed)
    elif data.dtype == np.complex64:
        randnoisereal = np.complex64(random_normal(len(data), dtype=np.float32, seed=seed))
        randnoiseimag = np.complex64(random_normal(len(data), dtype=np.float32, seed=seed))
        randnoise = randnoisereal + (1j * randnoiseimag)  # type: ignore[assignment]
    else:
        raise NotImplementedError("Currently, this only works for these dtypes: float32, complex64.")
    assert randnoise.dtype == data.dtype
    result = data + amplitude * randnoise
    assert result.dtype == data.dtype
    return result


@typechecked
def make_fft_positive_freqs_only(sig: NDArray, samp_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes an fft, returns only the positive frequencies.
    Return value is a tuple: (sample_freqs, fft_mag).
    `sample_freqs` ranges from 0 to approximately samp_rate/2

    >>> from pcdr._internal.basictermplot import plot
    >>> from pcdr import make_wave, make_fft_positive_freqs_only
    >>> samp_rate = 50
    >>> timestamps, wave = make_wave(samp_rate, 5, "complex", seconds=2)
    >>> sample_freqs, fft_mag = make_fft_positive_freqs_only(wave, samp_rate)
    >>> plot(sample_freqs, fft_mag, youtputsize=4)
    xmin: 0.00
    xmax: 24.50
    ymin: 0.00
    ymax: ...
    ~██████████o███████████████████████████████████████
    ~██████████████████████████████████████████████████
    ~█████████o█o██████████████████████████████████████
    ~ooooooooo███oooooooooooooooooooooooooooooooooooooo

    >>> timestamps, wave = makeWave(samp_rate, 5, "complex", seconds=0.2)
    >>> sample_freqs, fft_mag = make_fft_positive_freqs_only(wave, samp_rate)
    >>> plot(sample_freqs, fft_mag, youtputsize=4)
    xmin: 0.00
    xmax: 20.00
    ymin: 0.06
    ymax: ...
    ~█o███
    ~█████
    ~o█o██
    ~███oo
    """
    sample_freqs, fft_mag = make_fft(sig, samp_rate)
    halfway = len(sample_freqs) // 2
    return sample_freqs[halfway:], fft_mag[halfway:]


@typechecked
def make_fft(sig: np.ndarray, samp_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a tuple of (sample_freqs, fft_mag).
    `sample_freqs` ranges from approximately `-samp_rate/2` to approximately `samp_rate/2`.

    >>> from pcdr._internal.basictermplot import plot
    >>> from pcdr import make_wave, make_fft
    >>> seconds = 0.2
    >>> samp_rate = 50
    >>> freq = 5
    >>> timestamps, wave = make_wave(samp_rate, freq, "complex", seconds=seconds)
    >>> sample_freqs, fft_mag = make_fft(wave, samp_rate)
    >>> sample_freqs
    array([-25., -20., -15., -10.,  -5.,   0.,   5.,  10.,  15.,  20.])

    >>> plot(sample_freqs, fft_mag, youtputsize=2)
    xmin: -25.00
    xmax: 20.00
    ymin: 0.00
    ymax: ...
    ~██████o███
    ~oooooo█ooo

    >>> seconds = 0.8
    >>> samp_rate = 10
    >>> freq = 2
    >>> timestamps, wave = makeComplexWave_time(seconds, samp_rate, freq)
    >>> sample_freqs, fft_mag = make_fft(wave, samp_rate)
    >>> sample_freqs
    array([-5.  , -3.75, -2.5 , -1.25,  0.  ,  1.25,  2.5 ,  3.75])

    >>> plot(sample_freqs, fft_mag, youtputsize=2)
    xmin: -5.00
    xmax: 3.75
    ymin: 0.01
    ymax: ...
    ~██████o█
    ~oooooo█o

    Notice that the lowest sample frequency is not necessarily `samp_rate / 2`:
    >>> seconds = 0.9
    >>> samp_rate = 10
    >>> freq = 2
    >>> timestamps, wave = makeComplexWave_time(seconds, samp_rate, freq)
    >>> sample_freqs, fft_mag = make_fft(wave, samp_rate)
    >>> sample_freqs
    array([-4.44444444, -3.33333333, -2.22222222, -1.11111111,  0.        ,
            1.11111111,  2.22222222,  3.33333333,  4.44444444])
    """
    windowed = sig * np.hamming(len(sig))
    fft_result = np.fft.fftshift(np.fft.fft(windowed))
    sample_freqs = np.fft.fftshift(np.fft.fftfreq(len(windowed), 1/samp_rate))
    fft_mag = abs(fft_result)
    return sample_freqs, fft_mag
