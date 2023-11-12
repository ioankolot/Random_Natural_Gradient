import numpy as np
from quantum_circuit_exact import Qcir_Exact
import random
from fisher_information_matrix_circuit import Quantum_Fisher_Statevector, Classical_Fisher_Statevector
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector



class Statevector_Optimizers():
    def __init__(self, number_of_qubits,  layers, angles, maxiter, ansatz, single_qubit_gates, entanglement_gates, entanglement, problem):
        self.angles = angles.copy()
        self.maxiter = maxiter
        self.number_of_qubits = number_of_qubits
        self.layers = layers

        self.problem = problem['type']
        self.adjacency_matrix = problem['properties']
        self.offset = self.get_offset()

        
        self.ansatz = ansatz
        self.number_of_parameters = len(angles)
        self.single_qubit_gates = single_qubit_gates
        self.entanglement_gates = entanglement_gates
        self.entanglement = entanglement


    def get_offset(self):
        offset = 0
        for i in range(self.number_of_qubits):
            for j in range(self.number_of_qubits):
                if i<j:
                    offset += self.adjacency_matrix[i,j]/2
        return offset

    
    def sigma(self, z):
        if z == 0:
            value = 1
        elif z == 1:
            value = -1
        return value
    
    def maxcut_cost(self, x, w):
        total_energy = 0
        x = [int(_) for _ in x]
        for i in range(self.number_of_qubits):
            for j in range(self.number_of_qubits):
                if i<j:
                    if w[i,j] != 0:
                        total_energy += w[i,j] * (self.sigma(x[i]) * self.sigma(x[j]))
        total_energy /= 2
        return total_energy


    def expectation_value(self, angles, use_shots=False, shots=None):


        Qcir = Qcir_Exact(self.number_of_qubits, angles, self.layers, self.ansatz, self.adjacency_matrix) if self.ansatz=='HVA' else Qcir_Exact(self.number_of_qubits, angles, self.layers, self.ansatz, self.single_qubit_gates, self.entanglement_gates, self.entanglement)
        expectation_value = 0

        statevector = Statevector.from_label('0'*self.number_of_qubits)
        statevector = statevector.evolve(Qcir.qcir).reverse_qargs() #We reverse the qubits so that qubit 1 appears first.

        if use_shots:
            samples = statevector.sample_counts(shots)
            expectation_value = np.sum([(samples[sample]/shots)*self.maxcut_cost(sample, self.adjacency_matrix) for sample in samples])


        else:
            probabilities_dictionary= statevector.probabilities_dict() #We get the probabilities dictionary
            for key in probabilities_dictionary.keys():
                expectation_value += probabilities_dictionary[key]*self.maxcut_cost(key, self.adjacency_matrix)

        return np.real(expectation_value - self.offset)


    def calculate_derivatives(self, angles, which_derivatives):
        
        if not self.ansatz == 'HVA':
            derivatives = np.zeros((len(which_derivatives),))

            for parameter in range(len(which_derivatives)):

                plus_shift, minus_shift = angles.copy(), angles.copy()
                plus_shift[which_derivatives[parameter]] += np.pi/2
                minus_shift[which_derivatives[parameter]] -= np.pi/2

                derivatives[parameter] += 1/2*self.expectation_value(plus_shift)
                derivatives[parameter] -= 1/2*self.expectation_value(minus_shift)

        else:
            derivatives = np.zeros((len(which_derivatives),))

            for parameter in range(len(which_derivatives)):

                plus_shift, minus_shift = angles.copy(), angles.copy()
                plus_shift[which_derivatives[parameter]] += 0.001
                minus_shift[which_derivatives[parameter]] -= 0.001

                derivatives[parameter] += self.expectation_value(plus_shift)/(2*0.001)
                derivatives[parameter] -= self.expectation_value(minus_shift)/(2*0.001)

        return derivatives


    def gradient_descent(self, eta=0.01):
    
        thetas = self.angles.copy()
        initial_exp_value = self.expectation_value(thetas)
        print('We begin the optimization using Gradient Descent')
        print(f'with initial expectation value {initial_exp_value}')
        exp_values = [initial_exp_value]

        for _ in range(self.maxiter):

            derivatives = self.calculate_derivatives(thetas, range(len(self.angles)))

            
            thetas = [thetas[i] - eta*derivatives[i] for i in range(len(self.angles))]
            exp_value = self.expectation_value(thetas)
            print(exp_value)
            exp_values.append(exp_value)

        return exp_values

    def natural_gradient_descent(self, eta=0.1, basis='random', random_basis_layers = 1, rcond=10**-4):

        thetas = self.angles.copy()
        initial_exp_value = self.expectation_value(thetas)
        print(f'We begin the optimization using Random Natural Gradient with {random_basis_layers} random layers.')
        print(f'with initial expectation value {initial_exp_value}')
        exp_values = [initial_exp_value]

        
        for iteration in range(self.maxiter):
            
            derivatives = self.calculate_derivatives(thetas, range(len(thetas)))
            CFIM = Classical_Fisher_Statevector(self.number_of_qubits, thetas, self.layers, ansatz = self.ansatz, basis=basis, random_basis_layers = random_basis_layers)
            classical_fisher_matrix = CFIM.classical_fisher_information_matrix(thetas, self.single_qubit_gates, self.entanglement_gates, self.layers, self.entanglement)
            
            inverse_cfim = np.linalg.pinv(classical_fisher_matrix, rcond=rcond, hermitian=True)
            new_thetas = thetas.copy()

            for j in range(len(thetas)):
                for k in range(len(thetas)):
                    new_thetas[j] -= eta*inverse_cfim[j, k]*derivatives[k]

            exp_value = self.expectation_value(new_thetas)

            if basis == 'random':
                if exp_value < initial_exp_value:
                    print(exp_value)
                    exp_values.append(exp_value)
                    initial_exp_value = exp_value
                    thetas = new_thetas
                
                else:
                    print(initial_exp_value)
                    exp_values.append(initial_exp_value)
            
            else:
                print(exp_value)
                exp_values.append(exp_value)
                initial_exp_value = exp_value
                thetas = new_thetas

        return exp_values

    def quantum_natural_gradient(self, eta=0.1, rcond = 10**-4, type = 'lin_comb_full'):
        
        thetas = self.angles.copy()
        initial_exp_value = self.expectation_value(thetas)
        print('We begin the optimization using Quantum Natural Gradient')
        print(f'with initial expectation value {initial_exp_value}')
        exp_values = [initial_exp_value]
        if eta != 'decreasing':
            current_eta = eta
        

        for iteration in range(self.maxiter):
            
            if eta == 'decreasing':
                current_eta = 1/(iteration+1)


            derivatives = self.calculate_derivatives(thetas, range(len(self.angles)))


            QFIM = Quantum_Fisher_Statevector(self.number_of_qubits, thetas, self.layers, self.number_of_parameters)
            quantum_fisher_information_matrix = QFIM.QFIM_qiskit(values = thetas, single_qubit_gates = self.single_qubit_gates, entanglement_gates = self.entanglement_gates,
                                                                type = type, layers = self.layers, entanglement = self.entanglement)
            
            inverse_qfim = np.linalg.pinv(quantum_fisher_information_matrix,rcond=rcond, hermitian=True)


            for i in range(len(self.angles)):
                for j in range(len(self.angles)):

                    thetas[i] -= current_eta*inverse_qfim[i,j]*derivatives[j]


            exp_value = self.expectation_value(thetas)
            print(exp_value)
            exp_values.append(exp_value)


        return exp_values
    

    def stochastic_quantum_natural_gradient(self, parameters_to_sample, rcond=10**-4, eta=0.01, gradient_parameters = 'all'):

        thetas = self.angles.copy()
        initial_exp_value = self.expectation_value(thetas)
        print('We begin the optimization using Stochastic-Coordinate Quantum Natural Gradient')
        print(f'with initial expectation value {initial_exp_value}')
        exp_values = [initial_exp_value]
        
        for _ in range(self.maxiter):

            indices = random.sample(range(len(self.angles)), parameters_to_sample)
            indices.sort()

            if gradient_parameters != 'all':
                derivatives = self.calculate_derivatives(thetas, indices)
                QFIM = Quantum_Fisher_Statevector(self.number_of_qubits, thetas, self.layers, self.number_of_parameters)
                reduced_qfim = QFIM.QFIM_alternate(thetas = thetas, which_indices = indices, single_qubit_gates = self.single_qubit_gates, entangled_gates = self.entanglement_gates, 
                                                                    entanglement = self.entanglement) 
            

            else:
                derivatives = self.calculate_derivatives(thetas, indices)
                QFIM = Quantum_Fisher_Statevector(self.number_of_qubits, thetas, self.layers, self.number_of_parameters)
                reduced_qfim = QFIM.QFIM_qiskit(values = thetas, single_qubit_gates=self.single_qubit_gates, entanglement_gates=self.entanglement_gates,
                                                layers = self.layers, entanglement=self.entanglement, which_parameters=indices) 
                

            inverse_reduced_qfim = np.linalg.pinv(reduced_qfim, rcond=rcond, hermitian=True)
            new_thetas = thetas.copy()


            for j in range(len(indices)):
                for k in range(len(indices)):
                    new_thetas[indices[j]] -= eta*inverse_reduced_qfim[j, k]*derivatives[k]

            exp_value = self.expectation_value(new_thetas)

            if exp_value < initial_exp_value:
                print(exp_value)
                exp_values.append(exp_value)
                initial_exp_value = exp_value
                thetas = new_thetas
            
            else:
                print(initial_exp_value)
                exp_values.append(initial_exp_value)
            
        return exp_values

