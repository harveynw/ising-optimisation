import matplotlib.pyplot as plt

from quantum_anneal import QuantumAnneal
from sim_anneal.plot import plot_energy
from sim_quantum_anneal.hamiltonian import HamiltonianSQA
from sim_quantum_anneal.plot import plot_energy_trotter_range
from sim_quantum_anneal.problems import hamming_weight_spike

N = 40
T = 0.01

ham = HamiltonianSQA(optimise=hamming_weight_spike, T=T)
sqa = QuantumAnneal(hamiltonian=ham,
                    N=N,
                    P=20,
                    T=T, T_pre=3.0, T_n_steps=50,
                    gamma_start=1.5, gamma_end=1e-8, gamma_n_steps=50)

energy, state, pre_history, sqa_history = sqa.simulate()

plt.plot(ham.contribution_opt, label='Optimise Func')
plt.plot(ham.contribution_j_field, label='-J_field*(trotter couplings)')
plt.title('Contributions to energy calculations')
plt.legend()
plt.show()

plt.title('J_field over time')
plt.plot(ham.history_j_field_coefficient)
plt.show()

print('Comparison Finished:')
print('SQA', hamming_weight_spike(state), state)

# plotting energy over time
plot_energy(energy_func=hamming_weight_spike, history=pre_history, method="Pre-Anneal")
plot_energy_trotter_range(energy_func=hamming_weight_spike, history=sqa_history, method="Simulated Quantum Annealing")
