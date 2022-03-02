import numpy as np
import random

from sim_anneal.anneal import Anneal
from sim_anneal.plot import plot_1d_anneal

# Bounds for solution
xmin, xmax = 0.01, 3.0

# Optimisation functions specified
energy = lambda x: np.sin(2*x) + np.cos(4*x+3) + 2  # Our toy landscape
neighbour = lambda x: np.clip(x + random.uniform(-1, 1), a_min=xmin, a_max=xmax)

# Start at x=0.8 and run for 100 iterations
simulation = Anneal(s_0=0.8,
                    k_max=1000,
                    neighbour_func=neighbour,
                    energy_func=energy)

solution, history = simulation.simulate()

print("FOUND:", solution)
# print(history)

# Plot as animation
plot_1d_anneal(energy_func=energy, xmin=xmin, xmax=xmax, history=history)
