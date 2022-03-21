import numpy as np
import random

from tqdm import tqdm
from typing import Callable, Any, List
from dataclasses import dataclass

# State type can be any object/number
State = Any


def default_temperature(k: int, k_max: int) -> float:
    return 1.0-k/k_max


def default_acceptance(e, e_dash, t: float) -> float:
    return 1.0 if e_dash < e else np.exp(-(e_dash - e)/t)


@dataclass
class Anneal:
    """
    Class for performing Simulated Annealing
    https://en.wikipedia.org/wiki/Simulated_annealing#Pseudocode
    """

    # Initial state
    s_0: Any
    # Max number of iterations
    k_max: int
    # Function for picking a random neighbour
    neighbour_func: Callable[[State], Any]
    # Energy function
    energy_func: Callable[[State], float]
    # Temperature function
    temperature_func: Callable[[int, int], float] = default_temperature
    # Acceptance function
    acceptance_func: Callable[[State, State, float], float] = default_acceptance

    def simulate(self) -> (State, List[Any]):
        s = self.s_0
        history = []

        for k in tqdm(range(self.k_max)):
            history.append(s)

            t = self.temperature_func(k, self.k_max)
            s_new = self.neighbour_func(s)

            p = self.acceptance_func(self.energy_func(s),
                                     self.energy_func(s_new),
                                     t)

            if p > random.uniform(0, 1):
                s = s_new

        return s, history



