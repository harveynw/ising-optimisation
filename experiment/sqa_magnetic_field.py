from pathlib import Path
from typing import List
from experiment.execute import execute_experiments, ExperimentResult
from sim_anneal.plot import plot_energy
from sim_quantum_anneal.hamiltonian import HamiltonianSQA
from sim_quantum_anneal.plot import plot_energy_trotter_range
from sim_quantum_anneal.problems import hamming_weight_spike
from sim_quantum_anneal.quantum_anneal import QuantumAnneal

import numpy as np

if __name__ == '__main__':

    def new_dimension(experiment_args: List, key: str, values: List) -> List:
        new_args = []
        for exp in experiment_args:
            for val in values:
                exp[key] = val
                new_args.append(exp)
        return new_args


    name = 'sqa_magnetic_field_full'

    # Execute the actual experiment
    if not Path('experiment/' + name).exists():

        N, T = 40, 0.01
        hamiltonian = HamiltonianSQA(optimise=hamming_weight_spike, T=T)
        default_args = {
            'hamiltonian': hamiltonian,
            'N': N,
            'P': 20,
            'T': T, 'T_pre': 3.0, 'T_n_steps': 50,
            'gamma_start': 1.5, 'gamma_end': 1e-8, 'gamma_n_steps': 1000
        }

        # Different values to test
        experiment_args = []

        # Over ambient temperature
        for ambient_T in np.linspace(start=0.01, stop=0.1, num=10):
            experiment_args.append({'T': ambient_T})

        # Over gamma_n_steps
        experiment_args = new_dimension(experiment_args, 'gamma_n_steps', [100, 500, 1000])

        # Over gamma_start
        experiment_args = new_dimension(experiment_args, 'gamma_start', np.arange(1.5, 10.0, 0.5))

        # Over gamma_end
        experiment_args = new_dimension(experiment_args, 'gamma_end', [1e-1, 1e-3, 1e-5, 1e-7])

        print(f'Commencing {len(experiment_args)}')

        for i, result in enumerate(execute_experiments(optimiser=QuantumAnneal,
                                                       experiments_args=experiment_args,
                                                       default_args=default_args,
                                                       n_repetitions=20)):
            result.save(name)
            print(f"Saved result {i + 1}/{len(experiment_args)}")

    # # Load it from file system and analyse
    # results: List[ExperimentResult] = ExperimentResult.load_all(group_name=name, optimiser_name='QuantumAnneal')
    #
    # for i, result in enumerate(results):
    #     print(f'Experiment {i}: {result.arguments}')
    #     energy_result = [hamming_weight_spike(r[1]) for r in result.result]
    #
    #     print('Min', min(energy_result))
    #     print('Average', sum(energy_result) / len(energy_result))
    #     print('Max', max(energy_result))
    #
    # # Analysing a particular result
    # exp_6 = results[6]
    # solved_run = exp_6.result[1]
    #
    # final_energy, min_state, pre_history, sqa_history = solved_run
    #
    # plot_energy(energy_func=hamming_weight_spike, history=pre_history, method="Pre-Anneal")
    # plot_energy_trotter_range(energy_func=hamming_weight_spike, history=sqa_history,
    #                           method="Simulated Quantum Annealing")
