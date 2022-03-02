import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Any, List


def plot_1d_anneal(energy_func: Callable[[Any], float], xmin: float, xmax: float, history: List[Any]):
    for state in history:
        x = np.linspace(xmin, xmax, num=200)
        y = np.array([energy_func(x_i) for x_i in x])

        plt.plot(x, y)
        plt.vlines(x=state, ymin=0, ymax=np.max(y)+1, colors=['r'])
        plt.show()
