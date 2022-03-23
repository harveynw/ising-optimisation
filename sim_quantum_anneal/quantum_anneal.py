import numpy as np
import random
from dataclasses import dataclass

from sim_quantum_anneal.hamiltonian import hamiltonian_sqa


@dataclass
class Quantum_Anneal:

    # Problem couplings
    J: np.ndarray

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

    def simulate(self):
        # Random initial spin
        z = np.zeros(shape=(self.N, 1))

        for t in np.linspace(start=self.T_pre, stop=self.T, num=self.T_n_steps):
            for k in range(len(z)):
                z = self.metropolis(state=z, spin_i=k, spin_trotter=1, field_strength=self.gamma_start, tau=t)

        Z = np.repeat(z, self.P, axis=1)

        for gamma in np.linspace(start=self.gamma_start, stop=self.gamma_end, num=self.gamma_n_steps):
            for i in range(self.P):
                for k in range(len(z)):
                    Z = self.metropolis(state=Z, spin_i=k, spin_trotter=i, field_strength=gamma, tau=self.T)

        energies = [self.energy(state=Z[:, i], field_strength=self.gamma_end) for i in range(self.P)]

        min_idx = np.argmin(energies)
        return energies[min_idx], Z[:, min_idx]

    def metropolis(self, state: np.ndarray, spin_i: int, spin_trotter: int, field_strength: float, tau: float):
        delta = self.energy(state=state, field_strength=field_strength)
        state[spin_i, spin_trotter] *= -1
        delta = np.abs(delta - self.energy(state=state, field_strength=field_strength))

        if delta > 0 or np.exp(delta / tau) > np.random.random():
            # Return flipped
            return state

        # Unflip
        state[spin_i, spin_trotter] *= -1
        return state

    def energy(self, state: np.ndarray, field_strength: float) -> float:
        return hamiltonian_sqa(state=state, J=self.J, field_strength=field_strength)

