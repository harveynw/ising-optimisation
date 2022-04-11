from pathlib import Path
from typing import List
from experiment.execute import execute_experiments, get_experiment_path, ExperimentResult
from sim_quantum_anneal.hamiltonian import HamiltonianSQA
from sim_quantum_anneal.problems import hamming_weight_spike
from sim_quantum_anneal.quantum_anneal import QuantumAnneal

import numpy as np

if __name__ == '__main__':

    name = 'sqa_magnetic_field'

    # Execute the actual experiment
    if not Path('experiment/' + name).exists():

        N, T = 40, 0.01
        hamiltonian = HamiltonianSQA(optimise=hamming_weight_spike, T=T)
        default_args = {
            'hamiltonian': hamiltonian,
            'N': N,
            'P': 20,
            'T': T, 'T_pre': 3.0, 'T_n_steps': 50,
            'gamma_start': 1.5, 'gamma_end': 1e-8, 'gamma_n_steps': 20
        }

        # Different values to test
        experiment_args = [{'gamma_start': gamma_start} for gamma_start in np.arange(1.5, 10.0, 0.5)]

        for result in execute_experiments(optimiser=QuantumAnneal,
                                          experiments_args=experiment_args,
                                          default_args=default_args,
                                          n_repetitions=20):
            result.save(name)

    # Load it from file system and analyse
    results: List[ExperimentResult] = ExperimentResult.load_all(group_name=name, optimiser_name='QuantumAnneal')

    energies_n = []
    for i, result in enumerate(results):
        print(f'Experiment {i}: {result.arguments}')
        energy_result = [hamming_weight_spike(r[1]) for r in result.result]

        print('Min', min(energy_result))
        print('Average', sum(energy_result)/len(energy_result))
        print('Max', max(energy_result))

    print(energies_n)
