import numpy as np

from typing import List
from dataclasses import dataclass
from tqdm import tqdm
from sim_quantum_anneal.hamiltonian import System, HamiltonianSQA, ensure_2d


@dataclass
class QuantumAnneal:

    # Problem
    hamiltonian: HamiltonianSQA

    # Number of spins (variables in our Ising Model)
    N: int
    # Number of Trotter Slices
    P: int

    # Ambient Temperature, pre-anneal temperature, pre-anneal step-size
    T: float
    T_pre: float
    T_n_steps: int

    # Start, final and steps for magnetic field strength
    gamma_start: float
    gamma_end: float
    gamma_n_steps: int

    def simulate(self, pre_anneal=True):
        pre_history = []  # history states during pre-anneal
        simulation_history = []  # history of states during simulation

        print('Begin:',
              f'QuantumAnneal(N={self.N}, P={self.P}, T_n_steps={self.T_n_steps}, gamma_n_steps={self.gamma_n_steps})')

        # Random initial spin
        z = np.random.choice([-1, 1], size=self.N)

        if pre_anneal:
            print('*** PREANNEAL ****')
            z, pre_history = self.perform_preanneal(z=z)

        Z: System = np.repeat(ensure_2d(z), self.P, axis=1)

        print('*** SIMULATION ****')
        for gamma in tqdm(np.linspace(start=self.gamma_start, stop=self.gamma_end, num=self.gamma_n_steps)):
            # chessboard spin update pattern
            for k, i in self.sequential_indices():
                Z = self.metropolis(state=Z, spin_i=k, spin_trotter=i, field_strength=gamma, tau=self.T)

            simulation_history.append(Z.copy())  # store state

        # energies of states in each slice
        energies = [self.hamiltonian.evaluate(state=Z[:, i], field_strength=self.gamma_end) for i in range(self.P)]
        min_idx = np.argmin(energies)  # index of minimum energy slice in Z

        return energies[min_idx], Z[:, min_idx], pre_history, simulation_history

    def perform_preanneal(self, z: System):
        history = []

        for t in tqdm(np.linspace(start=self.T_pre, stop=self.T, num=self.T_n_steps)):
            for k in range(len(z)):
                z = self.metropolis(state=ensure_2d(z), spin_i=k, spin_trotter=0, field_strength=self.gamma_start, tau=t)

            history.append(z.copy())  # store state

        return z, history

    def chess_pattern_indices(self) -> List[int]:
        indices = []
        for i in range(self.P):
            if i % 2 == 0:
                for k in range(0, self.N, 2):
                    indices += [(k, i)]
            else:
                for k in range(1, self.N, 2):
                    indices += [(k, i)]
        for i in range(self.P):
            if i % 2 == 0:
                for k in range(1, self.N, 2):
                    indices += [(k, i)]
            else:
                for k in range(0, self.N, 2):
                    indices += [(k, i)]
        return indices

    def sequential_indices(self) -> List[int]:
        indices = []
        for k in range(self.N):
            for i in range(self.P):
                indices += [(k, i)]
        return indices

    def metropolis(self, state: System, spin_i: int, spin_trotter: int, field_strength: float, tau: float):
        E = self.hamiltonian.evaluate(state=state, field_strength=field_strength)  # energy of state
        state[spin_i, spin_trotter] *= -1  # flip spin i
        E_dash = self.hamiltonian.evaluate(state=state, field_strength=field_strength)  # energy of new state
        delta = E - E_dash  # energy diff

        if delta > 0 or np.exp(delta / tau) > np.random.random():
            # print(E, E_dash, delta, "Flipped")
            # Return flipped
            return state

        # print(E, E_dash, delta, "Not flipped")

        # Unflip
        state[spin_i, spin_trotter] *= -1
        return state
