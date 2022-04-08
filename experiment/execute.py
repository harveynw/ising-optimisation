import io
import os
import pickle
import random
import string

from contextlib import redirect_stdout
from dataclasses import dataclass
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
    result: tuple

    def save(self):
        with open(os.path.join(get_experiment_path(self.optimiser_name), self.experiment_name + '.pickle'), 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(file):
        d = pickle.load(file)
        return ExperimentResult(**d)


def get_experiments_path():
    p = Path('experiment/experiments')
    if not p.exists():
        p.mkdir()
    return p


def get_experiment_path(name: str):
    p = get_experiments_path() / name
    if not p.exists():
        p.mkdir()
    return p


def execute_optimiser(kwargs):
    optimiser = kwargs.pop('optimiser')
    instance = optimiser(**kwargs)
    with redirect_stdout(io.StringIO()) as f:
        return_values = instance.simulate()
    stdout = f.getvalue()

    name = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    return ExperimentResult(experiment_name=name,
                            stdout=stdout,
                            arguments=kwargs,
                            optimiser_name=optimiser.__name__,
                            result=return_values)


def execute_experiments(optimiser, experiments_args: List[dict], default_args: dict):
    args = []
    for experiment in experiments_args:
        args.append({'optimiser': optimiser, **default_args, **experiment})

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(execute_optimiser, arg) for arg in args]
        for future in as_completed(futures):
            print('Completed task')
            yield future.result()




