import typing
import math
import pandas as pd
import numpy as np
from scipy import fftpack
from scipy.signal import unit_impulse, filtfilt

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

FS = 1000.0
T = 1.0/FS

def pb1() -> None:
    df = pd.read_csv("data.txt", header=None)
    data = df[0].values
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(data, "-")
    ax.set_ylabel("Signal")
    ax.set_xlabel("Sampling step")
    plt.savefig("pb1_plot.png")

def pb2() -> None:
    df = pd.read_csv("data.txt", header=None)
    signal = df[0].values
    N = len(signal)

    fft_signal = fftpack.fft(signal)
    norm_fft_signal = np.abs(fft_signal)
    freqs = fftpack.fftfreq(N, T)
    i = freqs > 0
    
    plt.figure(figsize=(10,5))
    plt.ylabel('Amplitude')
    plt.xlabel('frequency')
    plt.plot(freqs[i], norm_fft_signal[i])
    plt.savefig("pb2_fft.png")

def delta_fn(x: float, ref: float) -> float:
    if abs(x - ref) < 1e-4:
        return 1.0
    else:
        return 0.0

def pb3() -> None:
    df = pd.read_csv("data.txt", header=None)
    signal = df[0].values
    N = len(signal)

    fft_signal = fftpack.fft(signal)
    norm_fft_signal = np.abs(fft_signal)
    freqs = fftpack.fftfreq(N, T)
    i_positives = freqs > 0

    F0 = 100.0
    deltaF = 10.0
    r = 1.0 - math.pi * F0 / FS

    theta_0 = 2.0 * math.pi * F0 / FS
    r = 1.0 - math.pi * deltaF / FS
    b_0 = np.abs(1.0 + 2.0 * r * np.cos(theta_0) + r**2.0) / (2.0 * np.abs(1.0 + np.cos(theta_0)))
    
    # 3.2
    f_space = np.linspace(0.0, FS/2.0, N)
    magnitude_response = np.subtract(1.0, unit_impulse(N, int(F0)))
    plt.plot(f_space, magnitude_response)
    plt.ylabel('Magnitude Response')
    plt.xlabel('Frequency')
    plt.savefig("pb3_2_magnitude_response.png")
    plt.cla();plt.clf()

    # 3.3
    vec_b = [b_0, -2.0 * b_0 * np.cos(theta_0), b_0]
    vec_a = [1.0, -2.0 * r * np.cos(theta_0), r**2.0]
    def get_filtered_y(
            x: typing.List[float], y: typing.List[float],
            k: int, _vec_b: float = vec_b, _vec_a: float = vec_a
        ) -> float:
        m = len(_vec_b) - 1
        vec_input = []
        for i in range(m+1):
            try:
                vec_input.append(_vec_b[i]*x[k-i])
            except IndexError:
                pass
        n = len(_vec_a)
        vec_output = []
        for i in range(1, n+1):
            try:
                vec_output.append(_vec_a[i]*y[k-i])
            except IndexError:
                pass
        return sum(vec_input) - sum(vec_output)
    filtered_signal = []
    for k in range(N):
        filtered_signal_k = get_filtered_y(
            x = signal, y = filtered_signal, k = k
        )
        filtered_signal.append(filtered_signal_k)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(filtered_signal, "-")
    ax.set_ylabel('Filtered signal')
    ax.set_xlabel('Sampling step')
    plt.savefig("pb3_3_filtered_signal_difference.png")
    plt.cla();plt.clf()

    # 3.4
    X = fftpack.fft(signal)
    Y = np.multiply(magnitude_response, X)
    y = fftpack.ifft(Y)
    print(X, magnitude_response)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y, "-")
    ax.set_ylabel('Filtered signal')
    ax.set_xlabel('Sampling step')
    plt.savefig("pb3_4_filtered_signal_infreqdom.png")
    plt.cla();plt.clf()

    # 3.5
    y = filtfilt(vec_b, vec_a, signal)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y, "-")
    ax.set_ylabel('Filtered signal')
    ax.set_xlabel('Sampling step')
    plt.savefig("pb3_5_filtered_signal_filterBA.png")
    plt.cla();plt.clf()

if __name__ == "__main__":
    # pb1()
    # pb2()
    pb3()