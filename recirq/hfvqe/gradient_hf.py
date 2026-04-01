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
"""
An implementation of gradient based Restricted-Hartree-Fock

This uses a bunch of the infrastructure already used in the experiment
should only need an RHF_object.
"""
from typing import Optional, Union
import numpy as np
import scipy as sp

from recirq.hfvqe.circuits import rhf_params_to_matrix
from recirq.hfvqe.objective import RestrictedHartreeFockObjective


def rhf_func_generator(
        rhf_objective: RestrictedHartreeFockObjective, #the rhf objective object, we would have defined it as this point before this routine as called
        initial_occ_vec: Optional[Union[None, np.ndarray]] = None, #initial occupation vector ~ defaults to none
        get_opdm_func: Optional[bool] = False): #whether or not to return the one-particle density matrix
    """Generate the energy, gradient, and unitary functions.

    Args:
        rhf_objective: objective function object
        initial_occ_vec: (optional) vector for occupation numbers of the
            alpha-opdm
    Returns:
        functions for unitary, energy, gradient (in that order)
    """
    if initial_occ_vec is None: #if there is no initial occupation vector 
        initial_opdm = np.diag([1] * rhf_objective.nocc +
                               [0] * rhf_objective.nvirt) #build an initial one-particle density matrix / builds a diagonal matrix with 1s for occupied and 0s for virtual (along the diagonal)
    else:
        initial_opdm = np.diag(initial_occ_vec) #else is we specify one we construct the opdm from the initial occupation vector
        """
        [[1, 0, 0, 0],
         [0, 1, 0, 0],                       Example for 2 occupied + 2 virtual
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
        """
    def energy(params): #get energy function
        u = unitary(params) #builds the unitary from params
        final_opdm_aa = u @ initial_opdm @ np.conjugate(u).T #applies the unitary transformation to the initial opdm,  so U * rho * U^(dag) (dag - conjugate transpose / hermitian adjoint)
        """
        @ is matrix multiplication 
        U * rho * U^(dag)
        """
        tenergy = rhf_objective.energy_from_opdm(final_opdm_aa) #pass the rotated opdm to compute the energy expectation value
        return tenergy

    def gradient(params): #get the gradient function
        u = unitary(params)
        final_opdm_aa = u @ initial_opdm @ np.conjugate(u).T
        return rhf_objective.global_gradient_opdm(params, final_opdm_aa).real #this time pases the rotated opdm to get the gradient, .real discards imaginary components

    def unitary(params): #takes the parameter vector
        kappa = rhf_params_to_matrix(
            params,
            len(rhf_objective.occ) + len(rhf_objective.virt), rhf_objective.occ,
            rhf_objective.virt) #builds the anti-hermitian matrix kappa 
        #this represents a matrix where the occupied-virtual and virtual-occupied blocks are nonzero
        return sp.linalg.expm(kappa) #computes e^kappa / or the unitary matrix U -> so this is how we generate a unitary rotation for parameters

    def get_opdm(params):
        u = unitary(params)
        return u @ initial_opdm @ np.conjugate(u).T #helper function to return the transformed opdm for given params

    if get_opdm_func: #returns the functions if get_opdm_func=True
        return unitary, energy, gradient, get_opdm
    return unitary, energy, gradient


def rhf_minimization(rhf_object, method='CG', initial_guess=None, verbose=True):
    """Perform Hartree-Fock energy minimization.

    Args:
        rhf_object: RestrictedHartreeFockObject we construct 
        method: scipy.optimize.minimize method - > CG is conjugate gradient
        initial_guess: defaults to none
        verbose: printing progreess -> Truw

    Returns:
        Scipy optimize result
    """
    _, energy, gradient = rhf_func_generator(rhf_object) #generates the energy, gradient, and unitary functions (we discard the unitary function)
    if initial_guess is None: #checks if initial guess has not been specified
        init_guess = np.zeros(rhf_object.nocc * rhf_object.nvirt) #defaults to all zeros (vector length nocc * nvirt) -> in RHF optimization paramerters are the orbital rotation angles between occupied and virtual orbitals
    else:
        init_guess = initial_guess.flatten() #if we specify an initial guess .flatten() turns it into a 1d array

    return sp.optimize.minimize(energy,
                                init_guess,
                                jac=gradient,
                                method=method,
                                options={'disp': verbose}) #performs the energy minimization using scipy minimizer routine
