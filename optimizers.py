import numpy as np
import random
from fisher_matrices import Classical_Fisher_Information_Matrix, Quantum_Fisher_Information_Matrix
from qiskit.primitives import Estimator
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit.circuit import ParameterVector

class Optimizers():
    def __init__(self, number_of_qubits, layers, angles, maxiter, ansatz, single_qubit_gates, entanglement_gates, entanglement, problem):
        self.angles = angles.copy()
        self.maxiter = maxiter
        self.number_of_qubits = number_of_qubits
        self.layers = layers

        self.hamiltonian = problem['operator']
        self.offset = problem['offset']

        
        self.ansatz = ansatz
        self.number_of_parameters = len(angles)
        self.single_qubit_gates = single_qubit_gates
        self.entanglement_gates = entanglement_gates
        self.entanglement = entanglement
        
        if self.ansatz == 'HEA':
            self.quantum_circuit = TwoLocal(num_qubits=self.number_of_qubits, rotation_blocks=self.single_qubit_gates, entanglement_blocks=self.entanglement_gates,
                                       entanglement=self.entanglement, reps=self.layers)

        self.initial_exp_value = self.expectation_value(self.angles)


    def expectation_value(self, angles, shots=None):
        
        estimator = Estimator() if not shots else Estimator(options={'shots':shots})
        exp_value = estimator.run(self.quantum_circuit, self.hamiltonian, angles).result().values[0]

        return np.real(exp_value + self.offset)


    def calculate_derivatives(self, angles, which_derivatives, shots=None):
        
        estimator = Estimator() if not shots else Estimator(options={'shots':shots})
        gradient = ParamShiftEstimatorGradient(estimator)
        parameters = ParameterVector('theta', length=self.number_of_parameters)
        
        parameters_to_calculate_gradient = [parameters[k] for k in which_derivatives]
        
        quantum_circuit = self.quantum_circuit.assign_parameters(parameters)
        derivatives = gradient.run(quantum_circuit, self.hamiltonian, [angles], [parameters_to_calculate_gradient]).result().gradients[0]
        
        
        return derivatives


    def gradient_descent(self, eta=0.01, which_parameters = 'all', shots=None):
    
        thetas = self.angles.copy()
        print('We begin the optimization using Gradient Descent')
        print(f'with initial expectation value {self.initial_exp_value}')
        exp_values = [self.initial_exp_value]
        
        if which_parameters == 'all':
            which_parameters = range(self.number_of_parameters)

        for _ in range(self.maxiter):

            derivatives = self.calculate_derivatives(thetas, which_parameters, shots)
            thetas = [thetas[i] - eta*derivatives[i] for i in range(len(self.angles))]
            exp_value = self.expectation_value(thetas, shots)
            print(f'Gradient Descent: Iteration {_}, Expectation Value: {exp_value}')
            exp_values.append(exp_value)

        return exp_values
    
    
    def random_natural_gradient(self, eta=0.01, basis='random', random_basis_layers=2, rcond=10**-4, shots=None):
        
        thetas = self.angles.copy()
        print(f'We begin the optimization using Random Natural Gradient with {random_basis_layers} random layers.')
        print(f'with initial expectation value {self.initial_exp_value}')  
        exp_values = [self.initial_exp_value]
        initial_exp_value = self.initial_exp_value
        
        for iteration in range(self.maxiter):
            
            derivatives = self.calculate_derivatives(thetas, range(len(thetas)), shots)
            CFIM = Classical_Fisher_Information_Matrix(self.number_of_qubits, self.number_of_parameters)
            
            cfim = CFIM.construct_cfim(ansatz = self.ansatz, basis = basis, random_basis_layers = random_basis_layers, meas_parameters='random',
                                       parameters = thetas, single_qubit_gates= self.single_qubit_gates,
                                       entanglement_gates=self.entanglement_gates, entanglement= self.entanglement, reps = self.layers,  shots = shots)
        
            inverse_cfim = np.linalg.pinv(cfim, rcond=rcond, hermitian=True)
            new_thetas = thetas.copy()
            new_thetas -= eta*np.array(inverse_cfim).dot(derivatives)

            exp_value = self.expectation_value(new_thetas)

            if basis == 'random':
                if exp_value < initial_exp_value:
                    print(f'Random Natural Gradient: Iteration {iteration}, Expectation Value {exp_value}')
                    exp_values.append(exp_value)
                    initial_exp_value = exp_value
                    thetas = new_thetas
                
                else:
                    print(f'Random Natural Gradient: Iteration {iteration}, Expectation Value {initial_exp_value}')
                    exp_values.append(initial_exp_value)
            
            else:
                print(f'Random Natural Gradient: Iteration {iteration}, Expectation Value {exp_value}')
                exp_values.append(exp_value)
                initial_exp_value = exp_value
                thetas = new_thetas

        return exp_values

    def quantum_natural_gradient(self, eta=0.1, rcond = 10**-4):
        
        thetas = self.angles.copy()
        print('We begin the optimization using Quantum Natural Gradient')
        print(f'with initial expectation value {self.initial_exp_value}')
        exp_values = [self.initial_exp_value]

        for iteration in range(self.maxiter):
            

            derivatives = self.calculate_derivatives(thetas, range(len(self.angles)))

            QFIM = Quantum_Fisher_Information_Matrix(self.number_of_qubits, self.number_of_parameters)
            quantum_fisher_information_matrix = QFIM.QFIM(ansatz=self.ansatz, values = thetas, single_qubit_gates = self.single_qubit_gates, entanglement_gates = self.entanglement_gates,
                                                        layers = self.layers, entanglement = self.entanglement)
            
            inverse_qfim = np.linalg.pinv(quantum_fisher_information_matrix,rcond=rcond, hermitian=True)


            thetas -= eta*inverse_qfim.dot(derivatives)

            exp_value = self.expectation_value(thetas)
            print(f'Quantum Natural Gradient: Iteration {iteration}, Expectation Value {exp_value}')
            exp_values.append(exp_value)


        return exp_values
    

    def stochastic_quantum_natural_gradient(self, parameters_to_sample, rcond=10**-4, eta=0.01, shots=None):

        thetas = self.angles.copy()
        print('We begin the optimization using Stochastic-Coordinate Quantum Natural Gradient')
        print(f'with initial expectation value {self.initial_exp_value}')
        exp_values = [self.initial_exp_value]
        initial_exp_value = self.initial_exp_value
        
        for iteration in range(self.maxiter):

            indices = random.sample(range(len(self.angles)), parameters_to_sample)
            indices.sort()

            derivatives = self.calculate_derivatives(thetas, indices, shots)
            
            QFIM = Quantum_Fisher_Information_Matrix(self.number_of_qubits, self.number_of_parameters)
            reduced_qfim = QFIM.QFIM(self.ansatz, thetas, self.single_qubit_gates, self.entanglement_gates,self.layers, 
                                     self.entanglement,  indices, shots)

            inverse_reduced_qfim = np.linalg.pinv(reduced_qfim, rcond=rcond, hermitian=True)
            new_thetas = thetas.copy()

            for j in range(len(indices)):
                for k in range(len(indices)):
                    new_thetas[indices[j]] -= eta*inverse_reduced_qfim[j, k]*derivatives[k]

            exp_value = self.expectation_value(new_thetas)

            if exp_value < initial_exp_value:
                print(f'Stochastic-Coordinate Quantum Natural Gradient: Iteration {iteration}, Expectation Value {exp_value}')
                exp_values.append(exp_value)
                initial_exp_value = exp_value
                thetas = new_thetas
            
            else:
                print(f'Stochastic-Coordinate Quantum Natural Gradient: Iteration {iteration}, Expectation Value {initial_exp_value}')
                exp_values.append(initial_exp_value)
            
        return exp_values

