import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

B = 100
t = np.arange(start=-100, stop=100, step=0.01)
y = list(map(lambda x: math.sin(x)/x if x != 0 else 1, t))

plt.plot(t, y)
plt.show()