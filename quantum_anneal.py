import numpy as np
import random
from dataclasses import dataclass

def metropolis(s, j, tau):
    s_flip = s.copy(deep=True)
    s_flip[j] *= -1

    delta = E(s) - E(s_flip)

    if delta > 0 or np.exp(delta/tau) > np.random.random():
        return s_flip
    else:
        return s

        
def hamiltonian(J, h, sigma):
    # H = -ΣJσσ - Σhσ
    couplings = np.einsum('ij,i,j', J, sigma, sigma)
    external = np.einsum('i,i', h, sigma)
    return -couplings - external


@dataclass
class Quantum_Anneal:

    N: int
    P: int
    T: float
    T_pre: float
    T_step: float
    Gamma_start: float
    Gamma_end: float
    Gamma_step: float
    z: np.ndarray


    def simulate(self):

        for t in range(self.T_pre, self.T, self.T_step):
            for i in len(z):
                z = metropolis(z, i, t)

        Z = np.repeat(z, self.P)

        for gamma in range(self.Gamma_start, self.Gamma_end, self.Gamma_step):
            for i in range(self.P):
                for k in range(len(z)):
                    Z[i] = metropolis(Z[i], k, self.T)

        energies = [hamiltonian()]

        Z[np.argmin()]