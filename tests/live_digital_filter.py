import scipy.signal
from sklearn.metrics import mean_absolute_error as mae
import numpy as np

# https://www.samproell.io/posts/yarppg/yarppg-live-digital-filter/

class _LiveFilter:
    """Base class for live filters.
    """
    def process(self, x):
        # do not process NaNs
        #if np.isnan(x):
        #    return x

        return self._process(x)

    def __call__(self, x):
        return self.process(x)

    def _process(self, x):
        raise NotImplementedError("Derived class must implement _process")


class _LiveSosFilter(_LiveFilter):
    """Live implementation of digital filter with second-order sections.
    """
    def __init__(self, sos):
        """Initialize live second-order sections filter.

        Args:
            sos (array-like): second-order sections obtained from scipy
                filter design (with output="sos").
        """
        self.sos = sos

        self.n_sections = sos.shape[0]
        self.state = np.zeros((self.n_sections, 2))

    def _process(self, x: float | list[float] | np.ndarray) -> float | list[float] | np.ndarray:
        """Filter incoming data with cascaded second-order sections.
        """
        if isinstance(x, list):
            return [self._process(xi) for xi in x]
        elif isinstance(x, np.ndarray):
            return np.array([self._process(xi) for xi in x])


        for s in range(self.n_sections):  # apply filter sections in sequence
            b0, b1, b2, a0, a1, a2 = self.sos[s, :]

            # compute difference equations of transposed direct form II
            y = b0*x + self.state[s, 0]
            self.state[s, 0] = b1*x - a1*y + self.state[s, 1]
            self.state[s, 1] = b2*x - a2*y
            x = y  # set biquad output as input of next filter section.

        return y
    
class SOSFilter(_LiveSosFilter):
    """
    Live implementation of digital filter with second-order sections.
    Simplified version of the previous class with less parameters and easier initialization.
    """

    def __init__(self, Wn: float | tuple[float, float], fs: float,
                 btype: str = "lowpass"):
        """Initialize live second-order sections filter.

        Args:
            Wn (float | tuple[float, float]): Cutoff frequency of the filter. When passing tuple, the first frequency
                is the lower cutoff frequency and the second frequency is the upper cutoff frequency.
            fs (float): Sampling frequency of the input signal.
            btype (str): Type of the filter (lowpass, highpass, bandpass, bandstop).
        """
        sos = scipy.signal.iirfilter(4, Wn=Wn, fs=fs, btype=btype,
                                     ftype="butter", output="sos")
        super().__init__(sos)

    def reset(self):
        """Reset filter state."""
        self.state = np.zeros((self.n_sections, 2))


class MultiChannelSOSFilter(_LiveFilter):
    """Live implementation of digital filter with second-order sections for multiple channels.
    """
    def __init__(self, Wn: float | tuple[float, float], fs: float,
                 btype: str = "lowpass", n_channels: int = 1):
        """Initialize live second-order sections filter.
        Expects input data to be in the shape of (n_channels, n_samples).

        Args:
            Wn (float | tuple[float, float]): Cutoff frequency of the filter. When passing tuple, the first frequency
                is the lower cutoff frequency and the second frequency is the upper cutoff frequency.
            fs (float): Sampling frequency of the input signal.
            btype (str): Type of the filter (lowpass, highpass, bandpass, bandstop).
            n_channels (int): Number of channels.
        """
        self.filters = [SOSFilter(Wn, fs, btype) for _ in range(n_channels)]

    def _process(self, x: float | list[float] | np.ndarray) -> float | list[float] | np.ndarray:
        """
        Filter incoming data with cascaded second-order sections.
        Expects input data to be in the shape of (n_channels, n_samples).
        
        """
        if isinstance(x, list):
            assert len(x) == len(self.filters), "Number of channels must match the number of filters."
            return [filter(xi) for filter, xi in zip(self.filters, x)]

        elif isinstance(x, np.ndarray):
            assert x.shape[0] == len(self.filters), "Number of channels must match the number of filters."
            return np.array([filter(xi) for filter, xi in zip(self.filters, x)])

        return [filter(x) for filter in self.filters]

    def reset(self):
        """Reset filter state."""
        for filter in self.filters:
            filter.reset()

if __name__ == "__main__":

    np.random.seed(42)  # for reproducibility
    # create time steps and corresponding sine wave with Gaussian noise
    fs = 30  # sampling rate, Hz
    ts = np.arange(0, 5, 1.0 / fs)  # time vector - 5 seconds

    ys = np.sin(2*np.pi * 1.0 * ts)  # signal @ 1.0 Hz, without noise
    yerr = 0.5 * np.random.normal(size=len(ts))  # Gaussian noise
    yraw = ys + yerr

    # define lowpass filter with 2.5 Hz cutoff frequency
    sos = scipy.signal.iirfilter(4, Wn=2.5, fs=fs, btype="low",
                                ftype="butter", output="sos")
    y_scipy_sosfilt = scipy.signal.sosfilt(sos, yraw)

    live_sosfilter = _LiveSosFilter(sos)
    # simulate live filter - passing values one by one
    y_live_sosfilt = [live_sosfilter(y) for y in yraw]

    sosfilter = SOSFilter(2, fs)
    y_live_sosfilt = sosfilter(yraw)

    print(f"sosfilter error: {mae(y_scipy_sosfilt, y_live_sosfilt):.5g}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(ts, yraw, label="raw")
    plt.plot(ts, y_scipy_sosfilt, label="scipy.sosfilt")
    plt.plot(ts, y_live_sosfilt, label="live_sosfilter")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
