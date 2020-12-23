import math
import typing
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Setting
FS = 1
T = 1/float(FS)
M = 512
L = 2048
MIN_NOISE = -0.05
MAX_NOISE = 0.05
A = 0.01
D = 500

# Function
def get_f(k: int, fs: int = FS, m: int = M) -> float:
    _f = float(k * fs) / (2.0 * (float(M) - 1.0))
    return _f

def get_x(k: int, t: float = T) -> float:
    _x = np.sin(2.0 * math.pi * get_f(k) * float(k) * t, dtype=float)
    return _x

def get_noise(k: int) -> float:
    _rand = random.uniform(MIN_NOISE, MAX_NOISE)
    return _rand

def get_y(k: int, d: int = D, m: int = M) -> float:
    j = k - d
    if j >= 0 and j < m:
        _signal = A * get_x(k - d)
    else:
        _signal = 0.0
    _noise = get_noise(k)
    _output = _signal + _noise
    return _output

def get_cross_corr(k: int, x: typing.List[float], y: typing.List[float]) -> float:
    _m = len(x)
    _l = len(y)
    _sum = 0.0
    for i in range(_l):
        j = i-k
        if j >= 0 and j < _m: _sum += y[i] * x[j]
    _corr = _sum/float(_l)
    return _corr

def get_norm_cross_corr(k: int, x: typing.List[float], y: typing.List[float]) -> float:
    _cross_corr = get_cross_corr(k, x, y)
    
    _m = len(x)
    _l = len(y)
    _rxx_0 = get_cross_corr(0, x, x)
    _ryy_0 = get_cross_corr(0, y, y)
    _norm = math.sqrt((_m/_l)*_rxx_0*_ryy_0)
    return _cross_corr/_norm

if __name__ == "__main__":
    input_signal = [get_x(k) for k in range(1, M+1)]
    noise = [get_noise(k) for k in range(L)]
    received_signal = [get_y(k) for k in range(L)]
    
    corr = [get_cross_corr(k, x=input_signal, y=received_signal) for k in range(L)]
    norm_corr = [get_norm_cross_corr(k, x=input_signal, y=received_signal) for k in range(L)]

    fig, axs = plt.subplots(5, figsize=(14,10))
    axs[0].plot(input_signal)
    axs[0].set_title("Correlation between transmitted signal and received signal (attenuation factor a=%.2f)"%A)
    axs[0].set_ylabel("Input signal")
    axs[1].plot(noise)
    axs[1].set_ylabel("Noise")
    axs[2].plot(received_signal)
    axs[2].set_ylabel("Received signal")

    axs[3].plot(corr)
    axs[3].set_ylabel("Linear\n cross-correlation")
    axs[4].plot(norm_corr)
    axs[4].set_ylabel("Normalized linear\n cross-correlation")

    # plt.show()
    plt.savefig("transmitted_received_corr.png")