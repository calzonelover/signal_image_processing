import typing
import math
import numpy as np


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


FS = 1600
N = 1024
T = 1.0/float(FS)

INV_SQRT2 = 1.0/math.sqrt(2)

def x(k: int) -> float:
    _k = float(k)
    _x = (np.sin(400.0*math.pi*_k*T)**2 \
        * np.cos(300.0*math.pi*_k*T)**2) \
        + np.random.normal(0.0, INV_SQRT2)
    return _x

def X(i: int, _wn: complex) -> float:
    wn_vec = np.vectorize(lambda k: _wn**(i*k))(range(N))
    x_vec = np.vectorize(lambda k: x(k))(range(N))
    return np.dot(x_vec, wn_vec)

def pb51():
    wn = np.exp(complex(0, -2.0*math.pi/float(N)))
    Wn_vec = np.vectorize(lambda i: X(i, wn))(range(N))
    Sx_vec = np.abs(Wn_vec)**2.0 / float(N)
    plt.stem(range(N), Sx_vec)
    plt.ylabel("S(f)")
    plt.xlabel("f")
    plt.savefig("pb5_1.png")
    # plt.show()

def pb52():
    x_vec = np.vectorize(lambda k: x(k))(range(N))
    avg_pow_x = np.mean(np.square(x_vec))
    print("Average power of x(i): ", avg_pow_x)
    noise = np.random.normal(0.0, INV_SQRT2, size=N)
    avg_pow_noise = np.mean(np.square(noise))
    print("Average power of moise: ", avg_pow_noise)

if __name__ == "__main__":
    pb51()
    pb52()