import numpy as np

from itertools import product
from quantum_anneal import Quantum_Anneal

N = 100

# Random couplings
r = lambda: np.random.random()
J = np.zeros(shape=(N, N))
for i, j in product(range(N), range(N)):
    J[i, j] = r()*2-1 if r() < 0.5 and i != j else 0

sqa = Quantum_Anneal(J=J,
                     N=N,
                     P=10,
                     T=0.5, T_pre=2, T_n_steps=100,
                     gamma_start=0.01, gamma_end=1, gamma_n_steps=50)

energy, state = sqa.simulate()

print('Found', sqa.energy_no_field(state), state)


