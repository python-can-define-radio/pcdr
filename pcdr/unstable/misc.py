import random
from typing import Optional
from numpy.typing import NDArray
import numpy as np
from typeguard import typechecked
from pcdr.unstable import str_to_bin_list, ook_modulate, multiply_by_complex_wave, noisify



@typechecked
def generate_ook_modulated_example_data(noise: bool = False, message_delay: bool = False, text_source: Optional[str] = None) -> NDArray[np.complex64]:
    """
    Generate a file with the given `output_filename`.

    if `noise` is True, random noise will be added to the generated signal.
    if `message_delay` is True, there will be a pause before the meaningful data starts.
    if `text_source` is any string, a random sentence from it will be used as the message.
    
    Example usage:

    text_content = "These are some words, and more words. There are many words in a row in these sentences."
    generate_ook_modulated_example_data(text_source=text_content)
    """
    message = "This is an example message."
    if text_source == None:
        print(f"No text source file specified, so all generated files will contain the message '{message}'")
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
        fm2 = np.concatenate([
            np.zeros(random.randint(100, 1500), dtype=np.complex64),
            fully_modded.y
        ])
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
def generate_ook_modulated_example_file(output_filename: str, noise: bool = False, message_delay: bool = False, text_source: Optional[str] = None):
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
