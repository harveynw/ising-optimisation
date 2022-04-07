import io
import os
import sys

from contextlib import redirect_stdout
from pathlib import Path
from typing import List
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed


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
    return stdout, return_values


def mute():
    sys.stdout = open(os.devnull, 'w')


def execute_experiments(optimiser, experiments_args: List[dict], default_args: dict):
    args = []
    for experiment in experiments_args:
        args.append({'optimiser': optimiser, **default_args, **experiment})

    with ProcessPoolExecutor(max_workers=5, initializer=mute) as executor:
        futures = [executor.submit(execute_optimiser, arg) for arg in args]
        for future in as_completed(futures):
            print('Completed task')
            yield future.result()




