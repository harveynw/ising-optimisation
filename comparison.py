import numpy as np

from itertools import product
from sim_anneal.anneal import Anneal
from sim_quantum_anneal.quantum_anneal import QuantumAnneal
from sim_anneal.plot import plot_energy

# Variables in our problem
N = 100

# Random couplings and external force
r = lambda: np.random.random()
J = np.zeros(shape=(N, N))
for i, j in product(range(N), range(N)):
    J[i, j] = r()*2-1 if r() < 0.5 and i != j else 0

def hamiltonian(J, h, sigma):
    # H = -ΣJσσ - Σhσ
    couplings = np.einsum('ij,i,j', J, sigma, sigma)
    external = np.einsum('i,i', h, sigma)
    return -couplings - external


def neighbour(sigma):
    flip = np.ones(shape=(N,), dtype=int)
    flip[np.random.randint(low=0, high=N)] = -1
    return sigma * flip


# Begin anneal
simulation = Anneal(s_0=np.random.choice([-1, 1], size=N),
                    k_max=10000,
                    neighbour_func=neighbour,
                    energy_func=lambda s: hamiltonian(J, h=np.zeros(shape=(N,)), sigma=s))

sa_solution, sa_history = simulation.simulate()

sqa = QuantumAnneal(J=J,
                     N=N,
                     P=20,
                     T=0.1, T_pre=2, T_n_steps=100,
                     gamma_start=1, gamma_end=0.01, gamma_n_steps=20)

energy, state, pre_history, sqa_history = sqa.simulate()

print('Comparison Finished:')
print('SA', sqa.energy_no_field(sa_solution), state)
print('SQA', sqa.energy_no_field(state), state)

#print(sa_history[5])
#print(sqa_history[5])


# plotting energy over time
plot_energy(energy_func=lambda s: sqa.energy_no_field(state=s), history=sa_history, method="Simulated Annealing")
plot_energy(energy_func=lambda s: sqa.energy_no_field(state=s), history=pre_history, method="Pre-Anneal")
plot_energy(energy_func=lambda s: sqa.energy_no_field(state=s), history=sqa_history, method="Simulated Quantum Annealing")