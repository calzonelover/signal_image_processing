import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

delta_fn = lambda x: 1 if x == 0 else 0
step_fn = lambda x: 1 if x >= 0 else 0

get_h = lambda i: -5.0 * delta_fn(i) + (1.67 + 5.33*((-0.5)**i))

def get_y(k):
    result = 0.0
    for i in range(k):
        t1 = get_h(i)
        t2 = 10.0 * np.sin(0.1*math.pi*(k-i)) * step_fn(k-i)
        result += t1 * t2
    return result

dt = 1
k = np.arange(start=0, stop=40, step=dt, dtype=int)
x = np.multiply(
    10.0,
    np.sin(0.1*math.pi*k),
)

h = np.vectorize(get_h)(k)
y = np.vectorize(get_y)(k)

fig, axs = plt.subplots(3, sharex=True, figsize=(12,7))
axs[0].stem(k, x, use_line_collection=True)
axs[0].set_title("x (k)")
axs[1].stem(k, h, use_line_collection=True)
axs[1].set_title("h (k)")
axs[2].stem(k, y, use_line_collection=True)
axs[2].set_title("y (k)")

plt.show()