import numpy as np

from dataclasses import dataclass
from tqdm import tqdm
from sim_quantum_anneal.hamiltonian_spike import hamiltonian_sqa, System, hamiltonian_spike


@dataclass
class QuantumAnneal:

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

        pre_history = []  # history states during pre-anneal
        simulation_history = []  # history of states during simulation

        print('Begin:',
              f'QuantumAnneal(N={self.N}, P={self.P}, T_n_steps={self.T_n_steps}, gamma_n_steps={self.gamma_n_steps})')

        # Random initial spin
        z = np.random.choice([-1, 1], size=self.N)
        print('Start', z)

        print('*** PRE-ANNEAL ****')
        for t in tqdm(np.linspace(start=self.T_pre, stop=self.T, num=self.T_n_steps)):
            for k in range(len(z)):
                z = self.metropolis(state=z.reshape((self.N, 1)), spin_i=k, spin_trotter=0, field_strength=self.gamma_start, tau=t)

            pre_history.append(z)  # store state

        Z: System = np.repeat(z, self.P, axis=1)

        print('*** SIMULATION ****')
        for gamma in tqdm(np.linspace(start=self.gamma_start, stop=self.gamma_end, num=self.gamma_n_steps)):
            # chessboard spin update pattern
            for i in range(self.P):
                if i % 2 == 0:
                    for k in range(0, len(z), 2):
                        Z = self.metropolis(state=Z, spin_i=k, spin_trotter=i, field_strength=gamma, tau=self.T)
                else:
                    for k in range(1, len(z), 2):
                        Z = self.metropolis(state=Z, spin_i=k, spin_trotter=i, field_strength=gamma, tau=self.T)
            for i in range(self.P):
                if i % 2 == 0:
                    for k in range(1, len(z), 2):
                        Z = self.metropolis(state=Z, spin_i=k, spin_trotter=i, field_strength=gamma, tau=self.T)
                else:
                    for k in range(0, len(z), 2):
                        Z = self.metropolis(state=Z, spin_i=k, spin_trotter=i, field_strength=gamma, tau=self.T)

            # find and store minimum energy state amongst slices
            energies = [self.energy_no_field(state=Z[:, i]) for i in range(self.P)]
            min_idx = np.argmin(energies)
            simulation_history.append(Z[:, min_idx])

        energies = [self.energy(state=Z[:, i], field_strength=self.gamma_end) for i in range(self.P)]

        min_idx = np.argmin(energies)
        return energies[min_idx], Z[:, min_idx], pre_history, simulation_history

    def metropolis(self, state: System, spin_i: int, spin_trotter: int, field_strength: float, tau: float):
        E = self.energy(state=state, field_strength=field_strength)
        state[spin_i, spin_trotter] *= -1
        E_dash = self.energy(state=state, field_strength=field_strength)
        delta = E - E_dash

        #print("E:", E)
        #print("E':", E_dash)

        if delta > 0 or np.exp(delta / tau) > np.random.random():
            # Return flipped
            print("metropolis accepted", delta)
            return state

        # Unflip
        state[spin_i, spin_trotter] *= -1
        return state

    def energy(self, state: System, field_strength: float) -> float:
        return hamiltonian_sqa(state=state, T=self.T, field_strength=field_strength)

    def energy_no_field(self, state: System) -> float:
        return hamiltonian_spike(state=state)
