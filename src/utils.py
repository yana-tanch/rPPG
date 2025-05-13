import shutil
from pathlib import Path

import numpy as np
from scipy import signal, sparse


def make_dir(directory):
    directory = Path(directory)

    if directory.is_dir():
        shutil.rmtree(directory)

    directory.mkdir(parents=True, exist_ok=True)

    return directory


def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    
    D = sparse.spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    
    return filtered_signal


def calculate_fft_hr(ppg_signal, fps=30, detrend_lambda=100, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    b, a = signal.butter(1, [0.75 / fps * 2, 2.5 / fps * 2], btype="bandpass")
    
    ppg_signal = detrend(ppg_signal, detrend_lambda)
    ppg_signal = signal.filtfilt(b, a, ppg_signal)
            
    ppg_signal = np.expand_dims(ppg_signal, 0)
    n = ppg_signal.shape[1]
    N = 2 ** (n - 1).bit_length() # Calculate the nearest power of 2
    f_ppg, pxx_ppg = signal.periodogram(ppg_signal, fs=fps, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60

    return fft_hr