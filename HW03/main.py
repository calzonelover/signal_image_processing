import typing
import numpy as np
from scipy.signal import lfilter, resample, filtfilt
from scipy.io import wavfile

import sounddevice as sd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

AUDIO_FILE = 'sample.wav'
ALPHA = 0.9
D = 5000
N_RESAMPLE = 268237 # 50000
a = 1
b = []
b.append(1)
for i in range(D): b.append(0)
b.append(ALPHA)

def get_result_signal(k: int, y: typing.List[float], alpha: float = ALPHA, d: int = D):
    if k - d < 0 or k - d >= N_RESAMPLE:
        result_signal = y[k]
    else:
        result_signal = y[k] + alpha * y[k - d]
    return result_signal

def pb1():
    original_sampling_rate, data = wavfile.read(AUDIO_FILE)
    data = [block[0] for block in data]
    # sd.play(data, N_RESAMPLE)
    # status = sd.wait()
    # data = [np.sqrt(np.mean(np.square(block))) for block in data]
    print(data)
    print(original_sampling_rate, len(data))
    data = resample(data, N_RESAMPLE)
    return data

def pb2(y):
    _v = lfilter(b, a, y)
    return _v

def pb3(v):
    _y_new = lfilter(a, b, v)
    # sd.play(_y_new, N_RESAMPLE)
    # status = sd.wait()
    return y_new

if __name__ == "__main__":
    y = pb1()
    v = pb2(y)
    y_back = pb3(v)
    assert y == y_back