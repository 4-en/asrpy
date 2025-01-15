import numpy as np
from asrpy import asr_calibrate, asr_process
import logging

class RTASR:
    """
    Uses asrpy to perform ASR in real-time with variable input chunk sizes.
    Automatically stores states and applies them to the next chunk of data.

    Parameters
    ----------
    sfreq : int
        Sampling frequency of the incoming data
    channel_count : int
        Number of channels in the incoming data
    lookahead : float
        Lookahead in seconds (default is 0.25)
        Lookahead * sfreq samples will be used for processing,
        effectively delaying the output by lookahead seconds.
    auto_calibrate : bool
        Automatically calibrate the ASR algorithm using the first samples
        (default is True)
    auto_calibrate_len_sec : int
        Length of the data to use for automatic calibration
        (default is 60 seconds)
    """

    def __init__(self,
                 sfreq: int,
                 channel_count: int,
                 lookahead: float = 0.25,
                 auto_calibrate: bool = True,
                 auto_calibrate_len_sec: int = 60
    ):
        self._sfreq = sfreq
        self._channel_count = channel_count
        self._lookahead = lookahead
        self.auto_calibrate = auto_calibrate
        self._auto_calibrate_len_sec = auto_calibrate_len_sec

        # calibration data
        self._was_calibrated = False # flag to indicate if ASR was calibrated
        self._M = None # Mixing array
        self._T = None # Threshold matrix

        # asr states
        self._R = None
        self._Zi = None
        self._cov = None
        self._carry = None
        

        # buffer for calibration (need to keep auto_calibrate_len_sec of data before processing)
        self.input_buffer = np.zeros((self._channel_count, self._sfreq * self._auto_calibrate_len_sec))
        self._input_buffer_index = 0

    @property
    def was_calibrated(self):
        return self._was_calibrated

    def calibrate(self, data: np.ndarray):
        """
        Calibrate the ASR algorithm using the provided data
        
        Parameters
        ----------
        data : np.ndarray, shape=(channel_count, sample_count)
            Data to use for calibration.
            *zero-mean* (e.g., high-pass filtered) and reasonably clean EEG of not
            much less than 30 seconds (this method is typically used with 1 minute
            or more).
        """
        # TODO: add more parameters from asr_calibrate

        if data.shape[0] != self._channel_count:
            raise ValueError(f"Data should have {self._channel_count} channels, but has {data.shape[0]} channels")

        self._M, self._T = asr_calibrate(data, self._sfreq)
        self._was_calibrated = True

        print("ASR calibrated")

    def _circular_slice(self, array: np.ndarray, start: int, end: int) -> np.ndarray:
        # Ensure start and end are within bounds
        if not (0 <= start < array.shape[1] and 0 <= end < array.shape[1]):
            raise ValueError(f"Start and end indices must be within the range [0, {array.shape[1]})")

        if start <= end:
            return array[:, start:end]
        else:
            return np.concatenate((array[:, start:], array[:, :end]), axis=1)
        
    def _circular_slice_by_len(self, array: np.ndarray, start: int, length: int) -> np.ndarray:
        # Ensure start and end are within bounds
        if not (0 <= start < array.shape[1] and 0 < length <= array.shape[1]):
            raise ValueError(f"Start index must be within the range [0, {array.shape[1]}) and length must be greater than 0")

        end = (start + length) % array.shape[1]
        return self._circular_slice(array, start, end)

    def _auto_calibrate(self):
        """
        Automatically calibrate the ASR algorithm using the first samples.
        This is triggered when the input_buffer has enough data.
        """

        if self._input_buffer_index < self._sfreq * self._auto_calibrate_len_sec:
            # not enough data for calibration
            return

        
        self.calibrate(self.input_buffer)


    def _add_to_input_buffer(self, data: np.ndarray):
        """
        Adds data to the input_buffer
        
        Parameters
        ----------
        data : np.ndarray, shape=(channel_count, sample_count)
            Data to add to the buffer
        """
        data_len = data.shape[1]

        # add data to the buffer
        if self._input_buffer_index + data_len > self.input_buffer.shape[1]:
            # add data until full
            self.input_buffer[:, self._input_buffer_index:] = data[:, :self.input_buffer.shape[1] - self._input_buffer_index]

            self._input_buffer_index = self.input_buffer.shape[1]
        else:
            self.input_buffer[:, self._input_buffer_index:self._input_buffer_index+data_len] = data
            self._input_buffer_index += data_len

        if not self._was_calibrated and self.auto_calibrate:
            self._auto_calibrate()


    def process(self, data: np.ndarray) -> np.ndarray | None:
        """
        Process incoming data with ASR
        
        Parameters
        ----------
        data : np.ndarray, shape=(channel_count, sample_count)
            Data to process.
        
        Returns
        -------
        np.ndarray, shape=(channel_count, sample_count)
            Processed data.
            This will return the raw input data if ASR has not been calibrated yet or if there is not 
            enough data in the buffer. Otherwise, it will return the transformed data.
        """

        # ensure data has the correct shape
        if data.shape[0] != self._channel_count:
            raise ValueError(f"Data should have {self._channel_count} channels, but has {data.shape[0]} channels")
        
        # ignore silently for now, but could be useful for debugging
        if data.shape[1] == 0:
            # no data to process
            return None
        
        # add data to buffer
        if not self._was_calibrated and self.auto_calibrate:
            # ASR has not been calibrated yet
            self._add_to_input_buffer(data)

        if not self._was_calibrated:
            # ASR has not been calibrated yet
            if not self.auto_calibrate:
                logging.warning("ASR has not been calibrated yet. Returning raw data.")
            return data
        
        # process data
        # TODO: handle lookahead and return None / zeroes while waiting for enough data
        clean, state = asr_process(
            data, 
            self._sfreq,
            self._M, 
            self._T,
            lookahead=self._lookahead,
            mem_splits=1 if data.shape[1] < 1000 else 3,
            R=self._R,
            Zi=self._Zi,
            cov=self._cov,
            carry=self._carry,
            return_states=True
            )
        
        # update state
        self._R = state['R']
        self._Zi = state['Zi']
        self._cov = state['cov']
        self._carry = state['carry']


        if clean.shape[1] != data.shape[1]:
            print(f"Processed data has different length than input data: {data.shape[1]} -> {clean.shape[1]}")

        return clean