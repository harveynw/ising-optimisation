import matplotlib.pyplot as plt

from typing import Callable, List
from sim_quantum_anneal.hamiltonian import System


def plot_energy_trotter_min(energy_func: Callable[[System], float], history: List[System], method: str):
    # Plot the minimum energy of the trotter slices over time

    min_energy_history = []
    _, P = history[0].shape
    for i in range(len(history)):
        energies = [energy_func(history[i][:, k]) for k in range(P)]
        min_energy_history.append(min(energies))

    plt.plot(min_energy_history)
    plt.title(f'Min energy function of trotter slices during {method}')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')

    plt.show()