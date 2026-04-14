# Copyright 2020 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union
from copy import deepcopy
from itertools import product

import numpy as np
from scipy.linalg import expm

from openfermion import slater_determinant_preparation_circuit
import cirq

from recirq.hfvqe import util


def rhf_params_to_matrix(parameters: np.ndarray,
                         num_qubits: int,
                         occ: Optional[Union[None, List[int]]] = None,
                         virt: Optional[Union[None, List[int]]] = None):
    """Assemble variational parameters into a matrix.

    For restricted Hartree-Fock we have nocc * nvirt parameters. These are
    provided as a list that is ordered by (virtuals) \times (occupied) where
    occupied is a set of indices corresponding to the occupied orbitals w.r.t
    the Lowdin basis and virtuals is a set of indices of the virtual orbitals
    w.r.t the Lowdin basis.  For example, for H4 we have 2 orbitals occupied and
    2 virtuals:

    occupied = [0, 1]  virtuals = [2, 3]

    parameters = [(v_{0}, o_{0}), (v_{0}, o_{1}), (v_{1}, o_{0}),
                  (v_{1}, o_{1})]
               = [(2, 0), (2, 1), (3, 0), (3, 1)]

    You can think of the tuples of elements of the upper right triangle of the
    antihermitian matrix that specifies the c_{b, i} coefficients.

    coefficient matrix
    [[ c_{0, 0}, -c_{1, 0}, -c_{2, 0}, -c_{3, 0}],
     [ c_{1, 0},  c_{1, 1}, -c_{2, 1}, -c_{3, 1}],
     [ c_{2, 0},  c_{2, 1},  c_{2, 2}, -c_{3, 2}],
     [ c_{3, 0},  c_{3, 1},  c_{3, 2},  c_{3, 3}]]

    Since we are working with only non-redundant operators we know c_{i, i} = 0
    and any c_{i, j} where i and j are both in occupied or both in virtual = 0.
    """
    if occ is None:
        occ = range(num_qubits // 2)
    if virt is None:
        virt = range(num_qubits // 2, num_qubits) # 

    # check that parameters are a real array
    if not np.allclose(parameters.imag, 0):
        raise ValueError("parameters input must be real valued")

    kappa = np.zeros((len(occ) + len(virt), len(occ) + len(virt)))
    for idx, (v, o) in enumerate(product(virt, occ)):
        kappa[v, o] = parameters[idx].real
        kappa[o, v] = -parameters[idx].real
    return kappa


def generate_circuits_from_params_or_u(
        qubits: List[cirq.Qid], #our list of gridcubits we made earlier ~ qubit # = # spatial orbitals
        parameters: np.ndarray, #unitary matrix u
        nocc: int, #number of occupied orbitals
        return_unitaries: Optional[bool] = False, #flag for whether or not to return unitaries
        occ: Optional[Union[None, List[int]]] = None, #optional arg for occupied orbitals
        virt: Optional[Union[None, List[int]]] = None, #optional arg for virt orbitals
        clean_ryxxy: Optional[bool] = False):  # testpragma: no cover
    """Make the circuits required for the estimation of the 1-RDM

    Args:
        qubits: define the qubits in the memory
        parameters: parameters of the kappa matrix
        nocc: number of occupied orbitals
        return_unitaries:  Check if the user wants unitaries returned
        occ: List of occupied indices
        virt: List of virtual orbitals
        clean_ryxxy: Determine the type of Givens rotation synthesis to use
            Options are 1, 2, 3, 4.
    """

    num_qubits = len(qubits) #get qubit count -> spatial orbitals
    # determine if parameters is a unitary
    if len(parameters.shape) == 2:
        if parameters.shape[0] == parameters.shape[1]:
            unitary = parameters #we are checking the shape here, if already a square 2d matrix we assume it is the unitary matrix
    else:
        generator = rhf_params_to_matrix(parameters,
                                         num_qubits,
                                         occ=occ,
                                         virt=virt) #build kappa
        unitary = expm(generator) #e^kappa

    circuits = [] #empty list to collect circuit builds
    unitaries = [] #empty list to collect unitaries
    for swap_depth in range(0, num_qubits, 2): #loop increments by 2, 4 qubit H_2 -> 0, 2, 4
        fswap_pairs = util.generate_fswap_pairs(swap_depth, num_qubits) #each depth gives a different permutation of qubit ordering -> we are building a swap network, generates pairs that will be (state) swapped at x depth
        swap_unitaries = util.generate_fswap_unitaries(fswap_pairs, num_qubits) #converts pairs into unitary matrices (we are accounting for fermionic antisymmetry, -1 phase)
        shifted_unitary = unitary.copy() #make a copy of the unitary
        for uu in swap_unitaries: #for unitary in new pairs, left multiply with shifted_unitary -> composing permutation into orbital rotation
            shifted_unitary = uu @ shifted_unitary
        unitaries.append(shifted_unitary) #store the shifted unitary in the list at current depth
        matrix = shifted_unitary.T[:nocc, :] #transpose the shifted unitary, take first nocc (# occupied orbitals) rows -> the slater determinant matrix

        permuted_circuit = cirq.Circuit() #create an empty cirq circuit
        permuted_circuit += prepare_slater_determinant(qubits,
                                                       matrix.real.copy(),
                                                       clean_ryxxy=clean_ryxxy) #composes our list of operations /gates, apply it to matrix
        # prepare_slater_determinant -> decomposes the slater determinant matrix into a sequence of givens rotations -> prepares gates
        circuits.append(permuted_circuit) #append circuit from list 

    if return_unitaries: #if this is marked True, return the unitary list 
        return circuits, unitaries

    return circuits #else return our circuits, we will append measurements in a later step


def xxyy_basis_rotation(pairs, clean_xxyy=False):
    """Generate the measurement circuits for the basis rotation"""
    all_ops = [] #empty list for gate ops

    for a, b in pairs:
        if clean_xxyy:
            all_ops += [
                cirq.rz(-np.pi * 0.25).on(a), #rotate qubit a's phase 
                cirq.rz(np.pi * 0.25).on(b), #rotate qubit b's phase 
                cirq.ISWAP.on(a, b)**0.5 #apply parital swap to complete basis rotation (maps to Z so XY can be measured)
            ]
        else: #we ignore this part this is for google hardware
            all_ops += [
                cirq.rz(-np.pi * 0.25).on(a),
                cirq.rz(np.pi * 0.25).on(b),
                cirq.FSimGate(-np.pi / 4, np.pi / 24).on(a, b)
            ]
    return all_ops


def circuits_with_measurements(qubits, circuits,
                               clean_xxyy=False):  # testpragma: no cover
    """Append the appropriate measurements to each of the permutation circuits.
    """
    num_qubits = len(qubits) #our number of qubits, # of spatial orbitals/basis functions(depends on basis set)
    even_pairs = [
        qubits[idx:idx + 2] for idx in np.arange(0, num_qubits - 1, 2)
    ] #collects the even neighboring pairs starting from 0 -> for H_2 [[q0,q1], [q2,q3], [q4,q5]]
    odd_pairs = [qubits[idx:idx + 2] for idx in np.arange(1, num_qubits - 1, 2)] #odd numbered pairs starting from 1 [[q1,q2], [q3,q4]]
    measure_labels = ['z', 'xy_even', 'xy_odd'] #measurement types, z is the standard measurement for orbital occupancy, xy_even measures even pairs in xy basis after basis rotation, xy_odd for measuring odd pairs
    all_circuits_with_measurements = {label: {} for label in measure_labels} #initializes an nested empty dict, outer keys being measurement labels, inner keys circuit indices 
    for circuit_index in range(len(circuits)): #loops over the number of permuted circuits
        for _, label in enumerate(measure_labels): #loop over each measurement label
            circuit = deepcopy(circuits[circuit_index]) #make a strict copy, important otherwise you would be messing with the same circuit in memory
            if label == 'xy_even': #if this is an xy_even measurement
                circuit.append(xxyy_basis_rotation(even_pairs,
                                                   clean_xxyy=clean_xxyy),
                               strategy=cirq.InsertStrategy.EARLIEST) #append rotation basis gate as early in the circuit as possible (strategy)
            if label == 'xy_odd':
                circuit.append(xxyy_basis_rotation(odd_pairs,
                                                   clean_xxyy=clean_xxyy),
                               strategy=cirq.InsertStrategy.EARLIEST) #append rotation basis gate as early in the circuit as possible (strategy)
            circuit.append(cirq.Moment([cirq.measure(q) for q in qubits])) #append measurement gates to circuits, moment means we do this step at the same time
            all_circuits_with_measurements[label][circuit_index] = circuit #stores the circuit under its label and index
    return all_circuits_with_measurements #return the nested dictionary of circuits


def prepare_slater_determinant(qubits: List[cirq.Qid],
                               slater_determinant_matrix: np.ndarray,
                               clean_ryxxy: Optional[Union[bool, int]] = True):
    """High level interface to the real basis rotation circuit generator.

    Args:
        qubits: List of cirq.Qids denoting logical qubits
        slater_determinant_matrix: basis rotation matrix
        clean_ryxxy: Optional[True, 1, 2, 3, 4] for indicating an error model
            of the Givens rotation.

    Returns:
        Generator for circuit
    """
    circuit_description = slater_determinant_preparation_circuit(
        slater_determinant_matrix) #calls an OpenFermion method, decomposes nocc x n_spacial slater
    # determinant matrix into givens rotations
    yield (cirq.X(qubits[j]) for j in range(slater_determinant_matrix.shape[0]))
    #apply X (NOT) gates to first nocc (.shape[0]) qubits, 1 qubit per 1 orbital, initialize state
    for parallel_ops in circuit_description: #iterating through layers of the circuit
        for op in parallel_ops: 
            i, j, theta, phi = op #unpack individual givens rotation into 2 qubits indice i, j
            #rotation angle theta, phi is phase
            if not np.isclose(phi, 0):  # testpragma: no cover
                raise ValueError("unitary must be real valued only")
            #checks to see if phi is nonzero, suggests complex phases in the unitary (RHF -> real)
            #the following selects which gate implementation to use for the givens rotation
            if clean_ryxxy is True or clean_ryxxy == 1:
                yield ryxxy(qubits[i], qubits[j], theta)
            elif clean_ryxxy == 2:
                yield ryxxy2(qubits[i], qubits[j], theta)
            elif clean_ryxxy == 3:
                yield ryxxy3(qubits[i], qubits[j], theta)
            elif clean_ryxxy == 4:
                yield ryxxy3(qubits[i], qubits[j], theta)
            else:
                raise ValueError("Invalide clean_ryxxy value")

#yield keyword -> creates generator functions -> pauses execution, returns a value to the caller, resume from that point

"""
We use the idealized circuit implementation (ryxxy -> seq of pauli operations to implement the givens rotation)
"""

def ryxxy(a, b, theta):
    """Implements the givens rotation with sqrt(iswap).
    The inverse(sqrt(iswap)) is made with z before and after"""
    yield cirq.ISWAP.on(a, b)**0.5 #ISWAP ~ partially swaps two qubit states while adding a phase factor
    yield cirq.rz(-theta + np.pi).on(a) #rotates the phase of qubit a's component by theta, how much the orbital a mixes into the rotation
    yield cirq.rz(theta).on(b) #rotates the phase of qubit b's component by theta
    yield cirq.ISWAP.on(a, b)**0.5  #applies the partial swap again, resolving entanglement -> the interference between the a and b components implements the actual orbital mixing
    yield cirq.rz(np.pi).on(a) #phase rotation to clean up residual phase

"""
the following circuit contructions below we don't use, they are for google hardware error correction, and do a simimlar strategy
"""
def ryxxy2(a, b, theta):
    """
    Implement realistic Givens rotation considering the always on parasitic
    cphase
    """
    yield cirq.FSimGate(-np.pi / 4, np.pi / 24).on(a, b)
    yield cirq.rz(-theta + np.pi).on(a)
    yield cirq.rz(theta).on(b)
    yield cirq.FSimGate(-np.pi / 4, np.pi / 24).on(a, b)
    yield cirq.rz(np.pi).on(a)


def ryxxy3(a, b, theta):
    """
    Implement realistic Givens rotation considering the always on parasitic
    cphase and attempt to reduce the error by 1/3
    """
    yield cirq.FSimGate(-np.pi / 4, np.pi / 24).on(a, b)
    yield cirq.rz(-theta + np.pi + np.pi / 48).on(a)
    yield cirq.rz(theta + np.pi / 48).on(b)
    yield cirq.FSimGate(-np.pi / 4, np.pi / 24).on(a, b)
    yield cirq.rz(np.pi + np.pi / 48).on(a)
    yield cirq.rz(+np.pi / 48).on(b)


def ryxxy4(a, b, theta):
    """
    Implement realistic Givens rotation considering the always on parasitic
    cphase and attempt to reduce the error by 1/3 for running on hardware
    """
    yield cirq.FSimGate(-np.pi / 4, 0).on(a, b)
    yield cirq.rz(-theta + np.pi + np.pi / 48).on(a)
    yield cirq.rz(theta + np.pi / 48).on(b)
    yield cirq.FSimGate(-np.pi / 4, 0).on(a, b)
    yield cirq.rz(np.pi + np.pi / 48).on(a)
    yield cirq.rz(+np.pi / 48).on(b)
