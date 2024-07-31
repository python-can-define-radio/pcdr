# pcdr

pcdr module: https://pypi.org/project/pcdr/

### Installation

```sh
pip install pcdr
```

Note that much of the functionality also depends on GNU Radio. However, `gnuradio` is not listed in the dependencies as it is not pip-installable (as far as the author knows).

### Example usage

```python3
from pcdr.unstable.flow import OsmoSingleFreqReceiver
tuned_freq = 103.9e6
receiver = OsmoSingleFreqReceiver("hackrf=0", tuned_freq)
receiver.start()
strength = receiver.get_strength()
print(f"Strength of {tuned_freq} Hz: {strength}")
receiver.stop_and_wait()
```
-->

### Current status

We launched version 1.0.0 with nothing marked as stable (i.e., everything marked as unstable). We admit that this is unconventional, but we hope that it will allow us to improve incrementally based on experimentation, and eventually (perhaps by early to mid 2025) begin to mark some parts as stable.

### Version history; breaking changes policy

In the v1.x series, we adhere to the following: 

- Everything in `pcdr._internal` should be ignored by users of this module.
- Everything in `pcdr.unstable` is in the experimentation phase and is subject to breaking changes between minor version updates.
- Everything in `pcdr.v0_compat` should be considered deprecated. As the name implies, it is kept for version 0 compatibility. It will not be changed.
- Everything else is stable. We will not intentionally make any breaking changes during the v1.x version series.

We made a significant amount of breaking changes when moving from version 0.x to version 1.0.0. If you have trouble migrating your code, please [submit an issue](https://github.com/python-can-define-radio/pcdr/issues), and we'd be happy to provide guidance.

### Repository history

The pcdr module's source used to be hosted as [a subfolder of the sdr-course repo](https://github.com/python-can-define-radio/sdr-course/tree/main/classroom_activities/Chx_Misc/Python_curric_2). On 2024 April 15, we migrated it to [its own repository](https://github.com/python-can-define-radio/pcdr). See the old repo if you'd like to refer to the commit history.
