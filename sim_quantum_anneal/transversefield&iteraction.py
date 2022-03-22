def TransverseFieldIsing(N,h):
  id = [[1, 0], [0, 1]]
  sigma_x = [[0, 1], [1, 0]]
  sigma_z = [[1, 0], [0, -1]]
  
  # vector of operators: [σᶻ, σᶻ, id, ...]
  first_term_ops = [None]*N
  first_term_ops[0] = sigma_z
  first_term_ops[1] = sigma_z
  for i in range(2,N):
    first_term_ops[i] = id
  
  # vector of operators: [σˣ, id, ...]
  second_term_ops = [None]*N
  second_term_ops[0] = sigma_x
  for i in range(1,N):
    second_term_ops[i] = id
  
  H = np.zeros((2^N, 2^N),  dtype=float, order='C')

  for i in range(1,N-1):
    # tensor multiply all operators
    # H -= foldl(⊗, first_term_ops)
    H -= np.tensordot(first_term_ops, 0)   # problem here
    # cyclic shift the operators
    first_term_ops = circshift(first_term_ops,1)
  
  for i in range(1,N):
    # H -= h*foldl(⊗, second_term_ops)
    H -= h * np.tensordot(second_term_ops, 0)
    second_term_ops = circshift(second_term_ops,1)   # problem here

  return H

TransverseFieldIsing(5,2)

from past.builtins import xrange
def GenerateNeighbors(nspins, 
                        J,  # scipy.sparse matrix
                        maxnb, 
                        savepath=None):
   
    # predefining vars
    ispin = 0
    ipair = 0
    # the neighbors data structure
    # cdef np.float_t[:, :, :]  
    nbs = np.zeros((nspins, maxnb, 2))
    # dictionary of keys type makes this easy
    J = J.todok()
    # Iterate over all spins
    for ispin in xrange(nspins):
        ipair = 0
        # Find the pairs including this spin
        for pair in J.keys():
            if pair[0] == ispin:
                nbs[ispin, ipair, 0] = pair[1]
                nbs[ispin, ipair, 1] = J[pair]
                ipair += 1
            elif pair[1] == ispin:
                nbs[ispin, ipair, 0] = pair[0]
                nbs[ispin, ipair, 1] = J[pair]
                ipair += 1
    J = J.tocsr()  # DOK is really slow for multiplication
    if savepath is not None:
        np.save(savepath, nbs)
    return nbs

# example for GenerateNeighbors
import numpy as np
from scipy.sparse import csr_matrix
  
row = np.array([0, 0, 1, 1, 2, 1])
col = np.array([0, 1, 2, 0, 2, 2])
  
# taking data
data = np.array([1, 4, 5, 8, 9, 6])
  
# creating sparse matrix
J = csr_matrix((data, (row, col)), 
                          shape = (3, 3))
GenerateNeighbors(3,J,5)

import numpy as np
from scipy.linalg import kron

spin_up = np.array([[1, 0]]).T
spin_down = np.array([[0, 1]]).T
# bit[0] = |0> = [1,0]', bit[1] = |1> = [0,1]'
bit = [spin_up, spin_down]


# construct more than two qubits
def basis(string='00010'):   # |00010>
    '''string: the qubits sequence'''
    res = np.array([[1]])
    
    for idx in string[::-1]:
        res = kron(bit[int(idx)], res)    
    return np.matrix(res)
