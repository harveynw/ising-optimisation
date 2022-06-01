# Ising Optimisation

An experimental library for using annealing algorithms on toy Ising model (and other!) optimisation problems.

### Contributors

[@dickonfell](https://github.com/dickonfell), [@JamieLNorth](https://github.com/JamieLNorth), [@harveynw](https://github.com/harveynw) and [@Joannazzh](https://github.com/Joannazzh). Developed during a project for the Edinburgh CAM MSc Program - supervised by Dr Matias Ruiz.

### Structure

- `/sim_anneal` - Code for performing <i>Simulated Annealing (SA)</i>, this is the popular classical algorithm. 
- `/sim_quantum_anneal` - Code for performing <i>Simulated Quantum Annealing (SQA)</i>, attempting to sample from the ground state of our problem hamiltonian using Trotter slices.
- `/experiment` - A sub-library for executing experiments over multiple CPU cores

### Setup
- Create a virtual environment
- `pip install -r requirements.txt`

### References

A crucial source in understanding and implementing Path-integral Monte Carlo for SQA:
- _Hadayat Seddiqi, pathintegral-qmc, (2017), [https://github.com/hadsed/pathintegral-qmc](https://github.com/hadsed/pathintegral-qmc)_