import typing
import math
import numpy as np


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


T0 = 1
F0 = float(1/T0)
A = 10.0
T = T0/float(5)
N = 128
M = 16

FS = 20
SAMPLING_FREQ = 1.0/float(FS)

def x(k: int) -> float:
    t = float(k) * SAMPLING_FREQ
    if t % T0 < T:
        return A
    return 0.0

def X(i: int, _wn: complex) -> float:
    wn_vec = np.vectorize(lambda k: _wn**(i*k))(range(N))
    x_vec = np.vectorize(lambda k: x(k))(range(N))
    return np.dot(x_vec, wn_vec)

def d(i: int, _wn: complex) -> float:
    return 2.0*np.abs(X(i, _wn))/float(N)

def phase(i: int, _wn: complex) -> float:
    return np.angle(X(i, _wn))

def xa(t: float, d_vec: typing.List[float], theta_vec: typing.List[float]) -> float:
    _xa = d_vec[0]/2.0
    for i in range(1,int(N/2)-1):
        _xa += d_vec[i] * np.cos(2.0*math.pi*i*t*F0 + theta_vec[i])
    return _xa

if __name__ == "__main__":
    i = [float(i) * SAMPLING_FREQ for i in range(N)]
    x_vec = np.vectorize(lambda k: x(k))(range(N))
    plt.plot(i, x_vec)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.savefig("pb4_x.png")
    plt.cla()
    plt.clf()
    # plt.show()

    wn = np.exp(complex(0, -2.0*math.pi/float(N)))
    X_vec = np.vectorize(lambda i: X(i, wn))(range(N))

    d_vec = [d(i, wn) for i in range(int(N/2))]
    plt.stem(d_vec)
    plt.xlabel("i")
    plt.ylabel(r"$d_i$")
    plt.savefig("pb4_d.png")
    plt.cla()
    plt.clf()

    phase_vec = [phase(i, wn) for i in range(int(N/2))]
    plt.stem(d_vec)
    plt.xlabel("i")
    plt.ylabel(r"$\theta_i$")
    plt.savefig("pb4_p.png")
    plt.cla()
    plt.clf()

    i = [float(i) * SAMPLING_FREQ for i in range(N)]
    xa_vec = [xa(t, d_vec, phase_vec) for t in i]
    plt.plot(i, xa_vec)
    plt.xlabel("i")
    plt.ylabel(r"$x_a$")
    plt.savefig("pb4_xa.png")
    plt.cla()
    plt.clf()