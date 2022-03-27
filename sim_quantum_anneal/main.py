import numpy as np

from itertools import product
from quantum_anneal import QuantumAnneal
from sim_quantum_anneal.hamiltonian import HamiltonianSQA
from sim_quantum_anneal.problems import ising_couplings

N = 10

# Random couplings
r = lambda: np.random.random()
J = np.zeros(shape=(N, N))
for i, j in product(range(N), range(N)):
    J[i, j] = r() * 2 - 1 if r() < 0.5 and i != j else 0

problem_func = lambda state: ising_couplings(J=J, state=state)

T = 0.5

sqa = QuantumAnneal(hamiltonian=HamiltonianSQA(optimise=problem_func, T=T),
                    N=N,
                    P=100,
                    T=T, T_pre=2, T_n_steps=100,
                    gamma_start=0.01, gamma_end=1, gamma_n_steps=50)

energy, state, _, _ = sqa.simulate()

print('Found', problem_func(state), state)
