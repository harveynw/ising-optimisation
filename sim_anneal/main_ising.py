import numpy as np

from itertools import product
from anneal import Anneal
from plot import plot_energy

# Size of our problem
n_vertices = 500

# Initial configuration {-1, 1} spin
sigma_0 = np.random.choice([-1, 1], size=n_vertices)

# Random couplings and external force
r = lambda: np.random.random()

J = np.zeros(shape=(n_vertices, n_vertices))
for i, j in product(range(n_vertices), range(n_vertices)):
    J[i, j] = r()*2-1 if r() < 0.5 and i != j else 0

h = np.zeros(shape=(n_vertices,))
for i in range(n_vertices):
    h[i] = r() * 2 - 1 if r() < 0.1 else 0


def hamiltonian(J, h, sigma):
    # H = -ΣJσσ - Σhσ
    couplings = np.einsum('ij,i,j', J, sigma, sigma)
    external = np.einsum('i,i', h, sigma)
    return -couplings - external


def neighbour(sigma):
    flip = np.ones(shape=(n_vertices,), dtype=int)
    flip[np.random.randint(low=0, high=n_vertices)] = -1
    return sigma * flip


# Our initial state
print(sigma_0)

# Begin anneal
simulation = Anneal(s_0=sigma_0,
                    k_max=1000,
                    neighbour_func=neighbour,
                    energy_func=lambda s: hamiltonian(J, h, s))

solution, history = simulation.simulate()

print("FOUND:", solution)

plot_energy(energy_func=lambda s: hamiltonian(J, h, s), history=history)
