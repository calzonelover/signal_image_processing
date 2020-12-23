import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def get_xa(_t, _T, _fs, _xs):
    _val = 0.0
    for n, xi in enumerate(_xs):
        _var = math.pi* _fs * (_t-n*_T)
        _val += xi * math.sin(_var)/_var if _var != 0 else xi
    return _val

f = 100
dt = 1e-5
t = np.arange(start=0, stop=0.1, step=dt)
x = np.sin(np.multiply(2.0*math.pi*f, t))


fs = 300
dT = 1.0/float(fs)
ts = np.arange(start=0, stop=0.1, step=dT)
xs = np.sin(np.multiply(2.0*math.pi*f, ts))

xa = list(map(lambda _t: get_xa(_t, _T=dT, _fs=fs, _xs=xs), t))


fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(12,7))
axs[0].plot(t, x, label="Signal")
axs[0].plot(ts, xs, "o", label="Sampling")
axs[0].plot(t, xa, "r--", alpha=0.4, label="Shannon Interpolation")
axs[0].legend()
axs[0].set_title("Over sampling")

fs = 70
dT = 1.0/float(fs)
ts = np.arange(start=0, stop=0.1, step=dT)
xs = np.sin(np.multiply(2.0*math.pi*f, ts))

xa = list(map(lambda _t: get_xa(_t, _T=dT, _fs=fs, _xs=xs), t))

axs[1].plot(t, x, label="Signal")
axs[1].plot(ts, xs, "o", label="Sampling")
axs[1].plot(t, xa, "r--", alpha=0.4, label="Shannon Interpolation")
# axs[1].legend()
axs[1].set_title("Under sampling")

plt.savefig("result_sin.png")
# plt.show()