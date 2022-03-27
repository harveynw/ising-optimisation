import numpy as np

from itertools import product
from sim_anneal.anneal import Anneal
from sim_quantum_anneal.quantum_anneal_spike import QuantumAnneal
from sim_anneal.plot import plot_energy

# Variables in our problem
N = 100

# Random couplings and external force
r = lambda: np.random.random()

def spike(state):
    w = sum((state + 1) / 2) # hamming weight of state vector
    n = state.shape[0] # length of state vector
    return n if w == n / 4 else w


def neighbour(sigma):
    flip = np.ones(shape=(N,), dtype=int)
    flip[np.random.randint(low=0, high=N)] = -1
    return sigma * flip


# Begin anneal
simulation = Anneal(s_0=np.random.choice([-1, 1], size=N),
                    k_max=100000,
                    neighbour_func=neighbour,
                    energy_func=lambda s: spike(state=s))

sa_solution, sa_history = simulation.simulate()

sqa = QuantumAnneal(N=N,
                     P=10,
                     T=5, T_pre=10, T_n_steps=50,
                     gamma_start=1.5, gamma_end=0.01, gamma_n_steps=20)

energy, state, pre_history, sqa_history = sqa.simulate()

print('Comparison Finished:')
print('SA', sqa.energy_no_field(sa_solution), state)
print('SQA', sqa.energy_no_field(state), state)

# plotting energy over time
plot_energy(energy_func=lambda s: spike(state=s), history=sa_history, method="Simulated Annealing")
plot_energy(energy_func=lambda s: spike(state=s), history=pre_history, method="Pre-Anneal")
plot_energy(energy_func=lambda s: spike(state=s), history=sqa_history, method="Simulated Quantum Annealing")
