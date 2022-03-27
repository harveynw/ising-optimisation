import numpy as np

from sim_quantum_anneal.hamiltonian import ensure_2d, System


def hamming_weight_spike(state: System) -> float:
    # spike cost function based on https://arxiv.org/pdf/quant-ph/0201031.pdf
    state = ensure_2d(state)

    n, P = state.shape
    summation = 0
    for k in range(P):
        w = sum((state[:, k] + 1) / 2) # hamming weight of state vector
        summation += n if w == n/4 else w

    return summation


def ising_couplings(J: np.ndarray, state: System) -> float:
    # This implementation is derived from:
    # https://github.com/hadsed/pathintegral-qmc/blob/master/docs/piqa_qmc_notes.pdf
    # This is the second inner summation term in Eqn. (2)
    state = ensure_2d(state)
    _, P = state.shape

    summation = 0
    for k in range(P):
        summation += np.einsum('ij,i,j', J, state[:, k], state[:, k])

    return summation
