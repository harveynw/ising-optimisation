import io
import os
import pickle
import random
import string

from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed


@dataclass
class ExperimentResult:
    experiment_name: str
    stdout: str
    arguments: dict
    optimiser_name: str
    result: list

    def save(self, group_name: str):
        # Persists this experiment result on disk under a group_name
        with open(os.path.join(get_experiment_path(group_name, self.optimiser_name),
                               self.experiment_name + '.pickle'), 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(file):
        # Loads a pickled ExperimentResult from disk
        d = pickle.load(file)
        return ExperimentResult(**d)

    @staticmethod
    def load_all(group_name: str, optimiser_name: str) -> List:
        # Loads multiple pickled ExperimentResults from disk
        path = get_experiment_path(group_name=group_name, optimiser_name=optimiser_name)
        results = []
        for p in path.rglob("*"):
            with p.open(mode='rb') as f:
                results += [ExperimentResult.load(f)]
        return results


def get_experiments_path(group_name: str):
    # Returns the path to the folder under a specific group_name
    p = Path('experiment/' + group_name)
    if not p.exists():
        p.mkdir()
    return p


def get_experiment_path(group_name: str, optimiser_name: str):
    # Returns the path to the folder under a group name and optimiser name.
    p = get_experiments_path(group_name) / optimiser_name
    if not p.exists():
        p.mkdir()
    return p


def execute_optimiser(kwargs, n_repetitions=1):
    # Calls .simulate on the 'optimiser' value n_repetitions times and returns the results
    optimiser = kwargs.pop('optimiser')
    instance = optimiser(**kwargs)
    with redirect_stdout(io.StringIO()) as f:
        return_values = [instance.simulate() for _ in range(n_repetitions)]
    stdout = f.getvalue()

    name = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    return ExperimentResult(experiment_name=name,
                            stdout=stdout,
                            arguments=kwargs,
                            optimiser_name=optimiser.__name__,
                            result=return_values)


def execute_experiments(optimiser, experiments_args: List[dict], default_args: dict, n_repetitions=1):
    # Executes each experiment in experiment_args (which overrides default_args) performing them n_repetitions times per
    # experiment using all CPU cores available.
    args = []
    for experiment in experiments_args:
        args.append({'optimiser': optimiser, **default_args, **experiment})

    task = partial(execute_optimiser, n_repetitions=n_repetitions)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(task, arg) for arg in args]
        for future in as_completed(futures):
            yield future.result()




