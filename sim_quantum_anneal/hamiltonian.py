import numpy as np

# This implementation is derived from:
# https://github.com/hadsed/pathintegral-qmc/blob/master/docs/piqa_qmc_notes.pdf

# Typing
System = np.ndarray


def hamiltonian_sqa(state: System, J: np.ndarray, T: float, field_strength: float) -> float:
    # This computes Eqn. (2)
    if state.ndim == 1:
        state = state.reshape((state.shape[0], 1))
    _, P = state.shape

    # Eqn. (3)
    J_field = -P*T/2.0 * np.log(np.tanh(field_strength/(P*T)))

    return -(hamiltonian_problem_couplings(state, J) +
             J_field*hamiltonian_trotter_couplings(state))


def hamiltonian_problem_couplings(state: System, J: np.ndarray) -> float:
    # This is the first inner summation term in Eqn. (2)
    if state.ndim == 1:
        state = state.reshape((state.shape[0], 1))
    _, P = state.shape

    summation = 0
    for k in range(P):
        summation += np.einsum('ij,i,j', J, state[:, k], state[:, k])

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
