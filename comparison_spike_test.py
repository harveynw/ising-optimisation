import numpy as np
import matplotlib.pyplot as plt

from sim_anneal.anneal import Anneal
from sim_anneal.plot import plot_energy

# Variables in our problem
from sim_quantum_anneal.hamiltonian import HamiltonianSQA
from sim_quantum_anneal.plot import plot_energy_trotter_range
from sim_quantum_anneal.problems import hamming_weight_spike
from sim_quantum_anneal.quantum_anneal import QuantumAnneal

N = 400

# Optional plotting of hamming weight 'spike' function
# bits = 100 # plz pick this to be congruent to 0 mod 4
# x = np.repeat(-1, bits)
# y = []
# for i in range(bits):
#     y.append(hamming_weight_spike(x))
#     x[i] = 1
# plt.title(f"Hamming Weight Spike for N={N} bit string")
# plt.plot(range(bits), y)
# plt.show()
# plt.savefig('spike.eps', format='eps')


# Random couplings and external force
def neighbour(sigma):
    flip = np.ones(shape=(N,), dtype=int)
    flip[np.random.randint(low=0, high=N)] = -1
    return sigma * flip


# SIMULATED ANNEALING
simulation = Anneal(s_0=np.random.choice([-1, 1], size=N),
                    k_max=100000,
                    neighbour_func=neighbour,
                    energy_func=hamming_weight_spike)

sa_solution, sa_history = simulation.simulate()


# SIMULATED QUANTUM ANNEALING
T = 0.1 # Ambient Temperature

sqa = QuantumAnneal(hamiltonian=HamiltonianSQA(optimise=hamming_weight_spike, T=T),
                    N=N,
                    P=10,
                    T=T, T_pre=10, T_n_steps=50,
                    gamma_start=100, gamma_end=10, gamma_n_steps=20)

energy, state, pre_history, sqa_history = sqa.simulate()


print('Comparison Finished:')
print('SA', hamming_weight_spike(sa_solution), state)
print('SQA', hamming_weight_spike(state), state)

# plotting energy over time
plot_energy(energy_func=hamming_weight_spike, history=sa_history, method="Simulated Annealing")
plot_energy(energy_func=hamming_weight_spike, history=pre_history, method="Pre-Anneal")
plot_energy_trotter_range(energy_func=hamming_weight_spike, history=sqa_history, method="Simulated Quantum Annealing")
