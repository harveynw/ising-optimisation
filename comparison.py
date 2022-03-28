import numpy as np

from itertools import product
from sim_anneal.anneal import Anneal
from sim_quantum_anneal.hamiltonian import HamiltonianSQA
from sim_quantum_anneal.problems import ising_couplings
from sim_quantum_anneal.quantum_anneal import QuantumAnneal
from sim_anneal.plot import plot_energy
from sim_quantum_anneal.plot import plot_energy_trotter_min, plot_energy_trotter_range

# Variables in our problem
N = 10

# Random couplings and external force
r = lambda: np.random.random()
J = np.zeros(shape=(N, N))
for i, j in product(range(N), range(N)):
    J[i, j] = r() * 2 - 1 if r() < 0.5 and i != j else 0


def neighbour(sigma):
    flip = np.ones(shape=(N,), dtype=int)
    flip[np.random.randint(low=0, high=N)] = -1
    return sigma * flip


problem_func = lambda state: ising_couplings(J=J, state=state)

# Begin anneal
simulation = Anneal(s_0=np.random.choice([-1, 1], size=N),
                    k_max=2000,
                    neighbour_func=neighbour,
                    energy_func=problem_func)

sa_solution, sa_history = simulation.simulate()

# Ambient temperature
T = 0.1

sqa = QuantumAnneal(hamiltonian=HamiltonianSQA(optimise=problem_func, T=T),
                    N=N,
                    P=20,
                    T=T, T_pre=2, T_n_steps=2,
                    gamma_start=1, gamma_end=0.01, gamma_n_steps=20)

energy, state, pre_history, sqa_history = sqa.simulate(pre_anneal=False)

print('Comparison Finished:')
print('SA', problem_func(sa_solution), sa_solution)
print('SQA', problem_func(state), state)

# plotting energy over time
plot_energy(energy_func=problem_func, history=sa_history, method="Simulated Annealing")
plot_energy(energy_func=problem_func, history=pre_history, method="Pre-Anneal")
plot_energy_trotter_min(energy_func=problem_func, history=sqa_history,
            method="Simulated Quantum Annealing")
plot_energy_trotter_range(energy_func=problem_func, history=sqa_history, method='SQA')
