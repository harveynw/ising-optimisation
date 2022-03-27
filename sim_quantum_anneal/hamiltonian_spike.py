import numpy as np

# This implementation is derived from:
# https://github.com/hadsed/pathintegral-qmc/blob/master/docs/piqa_qmc_notes.pdf

# Typing
System = np.ndarray


def hamiltonian_sqa(state: System, T: float, field_strength: float) -> float:
    # This computes Eqn. (2)
    if state.ndim == 1:
        state = state.reshape((state.shape[0], 1))
    _, P = state.shape

    # Eqn. (3)
    J_field = -P*T/2.0 * np.log(np.tanh(field_strength/(P*T)))

    return -(hamiltonian_spike(state) +
             J_field*hamiltonian_trotter_couplings(state))

def hamiltonian_spike(state: System) -> float:
    # spike cost function based on https://arxiv.org/pdf/quant-ph/0201031.pdf
    if state.ndim == 1:
        state = state.reshape((state.shape[0], 1))

    n, P = state.shape
    summation = 0
    for k in range(P):
        w = sum((state[:, k] + 1) / 2) # hamming weight of state vector
        summation += n if w == n/4 else w

    return summation

def hamiltonian_trotter_couplings(state: System) -> float:
    # This is the second inner summation term in Eqn. (2)
    if state.ndim == 1:
        state = state.reshape((state.shape[0], 1))
    N, P = state.shape

    summation = 0
    for k in range(P):
        for i in range(N):
            summation += state[i, k]*state[i, (k+1) % P]

    return summation
