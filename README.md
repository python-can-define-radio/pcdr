# pcdr

pcdr module: https://pypi.org/project/pcdr/

### Installation

```sh
pip install pcdr
```

Note that much of the functionality also depends on GNU Radio.
However, `gnuradio` is not listed in the dependencies as it is not
pip-installable (as far as the author knows).

### Example usage

```python3
from pcdr.flow import TODO
tuned_freq = 103.7e6
receiver = TODO("hackrf", freq=tuned_freq)
strength = receiver.get_strength()
print(f"Strength of {tuned_freq} Hz: {strength}")
receiver.stop_and_wait()

```
--->


### Repository history

The pcdr module's source was hosted as [a subfolder of the sdr-course repo](https://github.com/python-can-define-radio/sdr-course/tree/main/classroom_activities/Chx_Misc/Python_curric_2). On 2024 April 15, we migrated it to [its own repository](https://github.com/python-can-define-radio/pcdr). See the old repo if you'd like to refer to the commit history.

### Breaking changes

We made a significant amount of breaking changes when moving from version 0.x to version 1.0.0. If you have trouble migrating from a 0.x version, please [submit an issue](https://github.com/python-can-define-radio/pcdr/issues), and we'd be happy to provide guidance.
