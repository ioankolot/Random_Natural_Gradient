from qiskit.visualization import *
import numpy as np
from qiskit.quantum_info import Statevector, random_unitary
from qiskit.opflow import CircuitStateFn
from qiskit.opflow.gradients import QFI
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector
import scipy
import matplotlib.pyplot as plt
from qiskit.circuit.library import TwoLocal


class Quantum_Fisher_Statevector():
    def __init__(self, number_of_qubits, thetas, layers, number_of_parameters):

        self.number_of_qubits = number_of_qubits
        self.thetas = thetas.copy()
        self.layers = layers
        self.number_of_parameters = number_of_parameters

    
    def QFIM_HVA(self, values, adjacency_matrix, type='lin_comb_full'):
        quantum_circuit = QuantumCircuit(self.number_of_qubits)
        parameters = ParameterVector('theta', length=self.number_of_parameters)

        quantum_circuit.h(range(self.number_of_qubits))

        for layer in range(self.layers):

            for qubit1 in range(self.number_of_qubits):
                for qubit2 in range(self.number_of_qubits):
                    if qubit1<qubit2:
                        if adjacency_matrix[qubit1, qubit2] != 0:
                            quantum_circuit.rzz(parameters[2*layer], qubit1, qubit2)

            
            quantum_circuit.rx(parameters[2*layer+1], range(self.number_of_qubits))

        state = CircuitStateFn(primitive=quantum_circuit, coeff=1.)
        qfim = QFI(qfi_method=type).convert(operator=state, params=parameters)


        dictionary = {parameters[i]: values[i] for i in range(len(parameters))}
        qfim_result = qfim.assign_parameters(dictionary).eval() 

        return np.real(np.array(qfim_result))


    def QFIM_qiskit(self, values, single_qubit_gates, entanglement_gates, layers, entanglement, type='lin_comb_full', which_parameters = 'all'):
        quantum_circuit = QuantumCircuit(self.number_of_qubits)
        parameters = ParameterVector('theta', length=self.number_of_parameters)

        quantum_circuit += TwoLocal(num_qubits = self.number_of_qubits, rotation_blocks = single_qubit_gates, entanglement_blocks = entanglement_gates, 
                                    reps = layers, entanglement = entanglement)
        
        quantum_circuit = quantum_circuit.assign_parameters(parameters)

        state = CircuitStateFn(primitive=quantum_circuit, coeff=1.)

        qfim = QFI(qfi_method=type).convert(operator=state, params=parameters)

        dictionary = {parameters[i]: values[i] for i in range(len(parameters))}
        qfim_result = qfim.assign_parameters(dictionary).eval() 

        qfim = np.real(np.array(qfim_result))

        if which_parameters == 'all':

            return qfim
        
        else:

            for i in range(self.number_of_parameters):
                for j in range(self.number_of_parameters):
                    if i not in which_parameters or j not in which_parameters:
                        qfim[i,j] = 0

            return qfim
    
 

    def QFIM_alternate(self, thetas, which_indices, single_qubit_gates, entangled_gates, entanglement): 

        if which_indices == 'all':
            which_indices = range(len(thetas))

        two_local = TwoLocal(num_qubits=self.number_of_qubits, rotation_blocks=single_qubit_gates, entanglement_blocks=entangled_gates,
                                    reps = self.layers, entanglement=entanglement)
        
        two_local_inverse = two_local.copy().inverse()

        quantum_circuit = QuantumCircuit(self.number_of_qubits)
        original_thetas = self.thetas.copy()


        QFIM = np.zeros((len(which_indices), len(which_indices)))

        for parameter1 in range(len(which_indices)):
            for parameter2 in range(len(which_indices)):
                if parameter1 < parameter2:
                    
                    perturbed_thetas1, perturbed_thetas2, perturbed_thetas3, perturbed_thetas4 = self.thetas.copy(), self.thetas.copy(), self.thetas.copy(), self.thetas.copy()

                    perturbed_thetas1[which_indices[parameter1]] += np.pi/2
                    perturbed_thetas1[which_indices[parameter2]] += np.pi/2

                    perturbed_thetas2[which_indices[parameter1]] -= (np.pi/2)
                    perturbed_thetas2[which_indices[parameter2]] += (np.pi/2)

                    perturbed_thetas3[which_indices[parameter1]] += (np.pi/2)
                    perturbed_thetas3[which_indices[parameter2]] -= (np.pi/2)

                    perturbed_thetas4[which_indices[parameter1]] -= (np.pi/2)
                    perturbed_thetas4[which_indices[parameter2]] -= (np.pi/2)
                    
                    qcircuit1 = quantum_circuit.copy() + two_local.copy().assign_parameters(perturbed_thetas1) + two_local_inverse.copy().assign_parameters(original_thetas)
                    probabilities1 = Statevector.from_label('0'*self.number_of_qubits).evolve(qcircuit1).probabilities_dict()

                    qcircuit2 = quantum_circuit.copy() + two_local.copy().assign_parameters(perturbed_thetas2) + two_local_inverse.copy().assign_parameters(original_thetas)
                    probabilities2 = Statevector.from_label('0'*self.number_of_qubits).evolve(qcircuit2).probabilities_dict()

                    qcircuit3 = quantum_circuit.copy() + two_local.copy().assign_parameters(perturbed_thetas3) + two_local_inverse.copy().assign_parameters(original_thetas)
                    probabilities3 = Statevector.from_label('0'*self.number_of_qubits).evolve(qcircuit3).probabilities_dict()
                    
                    qcircuit4 = quantum_circuit.copy() + two_local.copy().assign_parameters(perturbed_thetas4) + two_local_inverse.copy().assign_parameters(original_thetas)
                    probabilities4 = Statevector.from_label('0'*self.number_of_qubits).evolve(qcircuit4).probabilities_dict()

                    try:
                        term1 = probabilities1['0'*self.number_of_qubits]
                    except:
                        term1 = 0
                
                    try:
                        term2 = probabilities2['0'*self.number_of_qubits]
                    except:
                        term2 = 0

                    try:
                        term3 = probabilities3['0'*self.number_of_qubits]
                    except:
                        term3 = 0

                    try:
                        term4 = probabilities4['0'*self.number_of_qubits]
                    except:
                        term4 = 0


                    QFIM[parameter1, parameter2] = -(term1 - term2 - term3 + term4)/8
                    
                elif parameter1==parameter2:

                    perturbed_thetas = self.thetas.copy()
                    perturbed_thetas[parameter1] += np.pi

                    qcircuit = quantum_circuit.copy() + two_local.copy().assign_parameters(perturbed_thetas) + two_local_inverse.copy().assign_parameters(original_thetas)
                    probabilities = Statevector.from_label('0'*self.number_of_qubits).evolve(qcircuit).probabilities_dict()
                    
                    try:
                        overlap = probabilities['0'*self.number_of_qubits]
                    except:
                        overlap = 0
                    term = (1-overlap)/4


                    QFIM[parameter1, parameter1] = term
                    
                else:
                    QFIM[parameter1, parameter2] = QFIM[parameter2, parameter1]


        return 4*QFIM


        
class Classical_Fisher_Statevector():
    def __init__(self, number_of_qubits, thetas, layers, ansatz = 'HEA', basis = 'z', random_basis_layers = 2, meas_parameters='random', adjacency_matrix = None):

        self.number_of_qubits = number_of_qubits
        self.thetas = thetas.copy() 
        self.layers = layers
        self.number_of_parameters = len(self.thetas)
        self.ansatz = ansatz
        self.basis = basis
        self.random_basis_layers  = random_basis_layers
        self.meas_parameters = [np.random.uniform(0, 2*np.pi) for _ in range((self.random_basis_layers+1)*self.number_of_qubits)] if meas_parameters == 'random' else meas_parameters
        self.basis_circuit = self.choose_measurement_basis()
        self.adjacency_matrix = adjacency_matrix



    def classical_fisher_information_matrix(self, parameters, single_qubit_gates, entangled_gates, reps, entanglement):

        classical_fisher = np.zeros((self.number_of_parameters, self.number_of_parameters))
        probabilities = self.get_probability_distribution(parameters, single_qubit_gates, entangled_gates, reps, entanglement)
        outcomes = probabilities.keys()

        all_derivatives = [self.calculate_outcomes_derivatives_alternative(parameter, single_qubit_gates, entangled_gates, reps, entanglement) for parameter in range(self.number_of_parameters)]


        for parameter1 in range(self.number_of_parameters):
            derivatives1 = all_derivatives[parameter1]

            for parameter2 in range(self.number_of_parameters):
                if parameter1 <= parameter2:
                    derivatives2 = all_derivatives[parameter2]

                    for outcome in outcomes:
                        classical_fisher[parameter1, parameter2] += (1/probabilities[outcome])*derivatives1[outcome]*derivatives2[outcome]
                
                else:
                    classical_fisher[parameter1, parameter2] = classical_fisher[parameter2, parameter1]


        return classical_fisher


    def calculate_outcomes_derivatives_alternative(self, parameter, single_qubit_gates, entangled_gates, reps, entanglement):

        derivatives_dictionary = {}
        angles_plus = self.thetas.copy()
        angles_minus = self.thetas.copy()

        if self.ansatz != 'HVA':
            angles_plus[parameter] += np.pi/2
            angles_minus[parameter] -= np.pi/2
        else:
            angles_plus[parameter] += 0.001
            angles_minus[parameter] -= 0.001

        probabilities_plus = self.get_probability_distribution(angles_plus, single_qubit_gates, entangled_gates, reps, entanglement)
        probabilities_minus = self.get_probability_distribution(angles_minus, single_qubit_gates, entangled_gates, reps, entanglement)

        outcomes = list(probabilities_plus.keys()) + list(set(probabilities_minus.keys()) - set(probabilities_plus.keys()))


        if self.ansatz != 'HVA':
            for outcome in outcomes:
                if outcome not in probabilities_plus.keys():
                    probabilities_plus[outcome] = 0
                if outcome not in probabilities_minus.keys():
                    probabilities_minus[outcome] = 0
                derivatives_dictionary[outcome] = (probabilities_plus[outcome] - probabilities_minus[outcome])/2

        else:
            for outcome in outcomes:
                if outcome not in probabilities_plus.keys():
                    probabilities_plus[outcome] = 0
                if outcome not in probabilities_minus.keys():
                    probabilities_minus[outcome] = 0
                derivatives_dictionary[outcome] = (probabilities_plus[outcome] - probabilities_minus[outcome])/(2*0.001)

        return derivatives_dictionary
    


    def get_probability_distribution(self, parameters, single_qubit_gates, entangled_gates, reps, entanglement):
        quantum_circuit = QuantumCircuit(self.number_of_qubits)

        if self.ansatz == 'HEA':
            quantum_circuit += TwoLocal(num_qubits = self.number_of_qubits, rotation_blocks = single_qubit_gates, entanglement_blocks = entangled_gates, reps = reps, entanglement = entanglement)
            quantum_circuit = quantum_circuit.assign_parameters(parameters)

        elif self.ansatz == 'HVA':
            
            quantum_circuit.h(range(self.number_of_qubits))

            for layer in range(self.layers):

                for qubit1 in range(self.number_of_qubits):
                    for qubit2 in range(self.number_of_qubits):
                        if qubit1<qubit2:
                            if self.adjacency_matrix[qubit1, qubit2] != 0:
                                quantum_circuit.rzz(parameters[2*layer], qubit1, qubit2)

            
                quantum_circuit.rx(parameters[2*layer+1], range(self.number_of_qubits))

        quantum_circuit.barrier()
        quantum_circuit += self.basis_circuit

        sv = Statevector.from_label('0'*self.number_of_qubits)
        sv = sv.evolve(quantum_circuit)
        probability_dictionary = sv.probabilities_dict()

        return probability_dictionary

    def add_random_measurement(self, meas_parameters, quantum_circuit, layers):


        for qubit in range(self.number_of_qubits):
            quantum_circuit.rx(meas_parameters[qubit], qubit)

        for layer in range(layers):
            for qubit1 in range(self.number_of_qubits):
                for qubit2 in range(self.number_of_qubits):
                    if qubit1<qubit2:
                        quantum_circuit.cz(qubit1, qubit2)

            for qubit in range(self.number_of_qubits):
                quantum_circuit.ry(meas_parameters[(layer+1)*self.number_of_qubits + qubit], qubit)


    def choose_measurement_basis(self):

        qcirc = QuantumCircuit(self.number_of_qubits)

        if self.basis == 'random_xyz':
            self.basis = np.random.choice(['x', 'y', 'z'])

        if self.basis == 'x':
            qcirc.h(range(self.number_of_qubits))

        elif self.basis == 'y':
            qcirc.rz(-np.pi/2, range(self.number_of_qubits))
            qcirc.h(range(self.number_of_qubits))

        elif self.basis == 'random':
            self.add_random_measurement(self.meas_parameters, qcirc, self.random_basis_layers)

        elif self.basis == 'haar_random':
            self.add_haar_random_unitary(qcirc)

        elif self.basis == 'random_pauli':
            for qubit in range(self.number_of_qubits):
                choice = np.random.choice(['x', 'y', 'z'])
                if choice == 'x':
                    qcirc.h(qubit)

                elif choice == 'y':
                    qcirc.rz(-np.pi/2, qubit)
                    qcirc.h(qubit)

        return qcirc
