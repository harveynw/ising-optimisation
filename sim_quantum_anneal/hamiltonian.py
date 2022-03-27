import numpy as np

from typing import Callable

# This implementation is derived from:
# https://github.com/hadsed/pathintegral-qmc/blob/master/docs/piqa_qmc_notes.pdf

# Typing
System = np.ndarray


def ensure_2d(state: System):
    # When a single trotter slice is passed, treat as system with P=1 slices.
    if state.ndim == 1:
        return state.reshape((state.shape[0], 1))
    else:
        return state


class HamiltonianSQA:
    def __init__(self, optimise: Callable[[System], float], T: float):
        # This is our problem to maximise, can be any function (linear or not)
        self.optimise = optimise

        # Ambient Temperature
        self.T = T

    def evaluate(self, state: System, field_strength: float):
        state = ensure_2d(state)
        _, P = state.shape

        # Eqn. (3)
        J_field = -P * self.T / 2.0 * np.log(np.tanh(field_strength / (P * self.T)))

        return -(self.optimise(state) + J_field*self.trotter_couplings_term(state))

    @staticmethod
    def trotter_couplings_term(state: System) -> float:
        # This is the second inner summation term in Eqn. (2)
        N, P = state.shape

        summation = 0
        for k in range(P):
            for i in range(N):
                summation += state[i, k] * state[i, (k + 1) % P]

        return summation
