import numpy as np

# This implementation is derived from:
# https://github.com/hadsed/pathintegral-qmc/blob/master/docs/piqa_qmc_notes.pdf


def hamiltonian_sqa(state: np.ndarray, J: np.ndarray, T: float, field_strength: float) -> float:
    # This computes Eqn. (2)
    _, P = state.shape

    # Eqn. (3)
    J_field = -P*T/2.0 * np.log(np.tanh(field_strength/(P*T)))

    return -(hamiltonian_problem_couplings(state, J) +
             J_field*hamiltonian_trotter_couplings(state))


def hamiltonian_problem_couplings(state: np.ndarray, J: np.ndarray) -> float:
    # This is the first inner summation term in Eqn. (2)
    _, P = state.shape

    summation = 0
    for k in range(P):
        summation += np.einsum('ij,i,j', J, state[:, k], state[:, k])

    return summation


def hamiltonian_trotter_couplings(state: np.ndarray) -> float:
    # This is the second inner summation term in Eqn. (2)
    N, P = state.shape

    summation = 0
    for k in range(P):
        for i in range(N):
            summation += state[i, k]*state[i, (k+1) % P]

    return summation
