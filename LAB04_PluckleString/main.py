import typing
import math
import numpy as np
from scipy.signal import stft

# import sounddevice as sd
# import soundfile as sf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

F0 = 740
fs = 44100.0
r = 0.999

def get_tf(z, l, c) -> float:
    denominator = 0.5*(c+(1.0+c)*z**(-1) + z**(-2))
    divider = 1.0 + c*z**(-1) - 0.5*(r**2)*(
        c*z**(-l) + (1.0 + c)*z**(-(l+1)) + z**(-(l+2))
    )
    return denominator/divider

def plot_transfer_function():
    L = math.floor((fs-0.5*F0)/F0)
    delta = (fs - (L+0.5)*F0)/F0
    c = (1-delta)/(1+delta)
    print(L, delta, c)

    n_sampling = 200
    f_min, f_max = 0.001, 25
    df = (f_max - f_min)/n_sampling
    fx = [f_min + i * df for i in range(n_sampling)]
    hz = list(map(lambda x: get_tf(x, L, c), fx))
    plt.plot(fx, hz)
    plt.show()


if __name__ == "__main__":
    plot_transfer_function()











