#   MULTIPROCESSING
#   experiment/sa_example.py
#   This file is used for performing grid search over different hyperparameters
#   of SA and SQA.
import numpy as np

from experiment.execute import get_experiments_path, execute_experiments, get_experiment_path, ExperimentResult
from itertools import product
from sim_anneal.anneal import Anneal

# *** BEGIN SA Problem Setup ***
# Size of our problem
n_vertices = 40
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
def energy(s):
    return hamiltonian(J, h, s)
# *** END ***


if __name__ == '__main__':
    name = 'sa_example'

    default_args = {
        's_0': sigma_0, 'k_max': 1000,
        'neighbour_func': neighbour,
        'energy_func': energy,
    }

    experiments_args = [
        {'k_max': 100},
        {'k_max': 1000},
        {'k_max': 10000},
        {'k_max': 100000},
    ]

    for result in execute_experiments(optimiser=Anneal, experiments_args=experiments_args, default_args=default_args,
                                      n_repetitions=10):
        result.save(name)

    results = ExperimentResult.load_all(group_name=name, optimiser_name='Anneal')
