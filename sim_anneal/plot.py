import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Any, List

from matplotlib.animation import FuncAnimation


def plot_1d_anneal(energy_func: Callable[[Any], float], xmin: float, xmax: float, history: List[Any]):
    n = len(history)

    # Create fig and ax
    fig, ax = plt.subplots()
    fig.suptitle('Anneal History', weight='bold')

    # Overlay our objective (energy) function
    x = np.linspace(xmin, xmax, num=200)
    y = np.array([energy_func(x_i) for x_i in x])
    ax.plot(x, y)

    state = ax.axvline(x=history[0], ymin=0, ymax=np.max(y)+1, color='r')

    def animate(i):
        ax.set_title(f'State x={history[i]:.2f}, Time {(i/n):.2%}')
        state.set_xdata(history[i])

    ani = FuncAnimation(fig, animate, frames=n, interval=10)
    plt.show()


def plot_energy(energy_func: Callable[[Any], float], history: List[Any], method: str):
    energy_history = [energy_func(state) for state in history]

    plt.plot(energy_history)
    plt.title(f'Energy function during {method}')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')

    plt.show()
