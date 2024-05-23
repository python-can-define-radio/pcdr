from pcdr.unstable.misc import (
    generate_ook_modulated_example_data,
    generate_ook_modulated_example_file,
    str_to_bin_list,
    ook_modulate,
    ook_modulate_at_frequency,
)


from pcdr._internal.wavegen import (
    make_wave,
    multiply_by_complex_wave,
    multiply_by_real_wave,
    make_fft,
    make_fft_positive_freqs_only,
    noisify,
)


from pcdr._internal.misc import (
    gnuradio_send
)

from pcdr._internal.basictermplot import plot


class AliasError(ValueError):
    pass


class NonBitError(ValueError):
    pass
