import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

delta_fn = lambda x: 1 if x == 0 else 0
step_fn = lambda x: 1 if x >= 0 else 0

def get_h(k: int) -> float:
    if k % 2 == 0:
        _h = 0.0
    else:
        if k % 3 == 0:
            _h = ((-0.5)**k - (0.5)**k)*step_fn(k)
        else:
            _h = -((-0.5)**k - (0.5)**k)*step_fn(k)
    return _h 


if __name__ == "__main__":
    ks = [k for k in range(0, 50)]
    h = [get_h(k) for k in ks]

    plt.stem(ks, h)
    plt.xlabel("k")
    plt.ylabel("h(k)")
    plt.title("Impulse Response")
    plt.savefig("impulse_response.png")
    plt.show()