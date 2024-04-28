import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, ParameterVector
import scipy
import matplotlib.pyplot as plt
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator, Sampler
from qiskit_algorithms.gradients import LinCombQGT, ParamShiftSamplerGradient


class Quantum_Fisher_Information_Matrix():
    def __init__(self, number_of_qubits, number_of_parameters):
        
        self.number_of_qubits = number_of_qubits
        self.number_of_parameters = number_of_parameters
        
    def QFIM(self, ansatz, values, single_qubit_gates, entanglement_gates, layers, entanglement, which_parameters = 'all', shots=None):
        
        parameters = ParameterVector('theta', length=self.number_of_parameters)
        
        if ansatz == 'HEA':
            
            quantum_circuit = TwoLocal(num_qubits = self.number_of_qubits, rotation_blocks = single_qubit_gates, entanglement_blocks = entanglement_gates,
                                       reps = layers, entanglement = entanglement)
            
        quantum_circuit = quantum_circuit.assign_parameters(parameters)
        
        estimator = Estimator() if not shots else Estimator(options={'shots':shots})
        QGT = LinCombQGT(estimator)

        if which_parameters == 'all':
            
            qfim = 4*np.real(QGT.run(quantum_circuit, [values]).result().qgts[0])
            
        else:
            parameters_to_calculate_qgt = [parameters[k] for k in which_parameters]
            qfim = 4*np.real(QGT.run(quantum_circuit, [values], [parameters_to_calculate_qgt]).result().qgts[0])
            
        return qfim

class Classical_Fisher_Information_Matrix():
    
    def __init__(self, number_of_qubits, number_of_parameters):

        self.number_of_qubits = number_of_qubits
        self.number_of_parameters = number_of_parameters
        
        

    def construct_cfim(self, ansatz='HEA', basis = 'random', random_basis_layers = 2, meas_parameters = 'random',
                       parameters=None, single_qubit_gates = None, entanglement_gates = None, entanglement = None, reps=None, shots=None):
        
        classical_fisher = np.zeros((self.number_of_parameters, self.number_of_parameters))
        meas_parameters = [np.random.uniform(0, 2*np.pi) for _ in range((random_basis_layers+1)*self.number_of_qubits)] if meas_parameters=='random' else meas_parameters
        
        if ansatz == 'HEA':
            
            quantum_circuit = TwoLocal(num_qubits=self.number_of_qubits, rotation_blocks=single_qubit_gates, entanglement_blocks=entanglement_gates,
                                       reps = reps, entanglement = entanglement)
            
        quantum_circuit &= self.choose_measurement_basis(basis, meas_parameters, random_basis_layers)
        quantum_circuit.measure_all()
        sampler = Sampler() if not shots else Sampler(options={'shots':shots})
        probabilities = sampler.run([quantum_circuit], [parameters]).result().quasi_dists[0]
        
        gradient = ParamShiftSamplerGradient(sampler)
        gradients_result = gradient.run([quantum_circuit], [parameters]).result().gradients[0]
        
        outcomes = range(2**self.number_of_qubits)
        
        for parameter1 in range(self.number_of_parameters):
            derivatives1 = gradients_result[parameter1]
            
            for parameter2 in range(self.number_of_parameters):
                if parameter1 <= parameter2:
                    derivatives2 = gradients_result[parameter2]
                    
                    for outcome in outcomes:
                        try:
                            classical_fisher[parameter1, parameter2] += (1/probabilities[outcome])*derivatives1[outcome]*derivatives2[outcome]
                        except:
                            pass
            
                else:
                    classical_fisher[parameter1, parameter2] = classical_fisher[parameter2, parameter1]
                    
        return classical_fisher

    def choose_measurement_basis(self, basis, meas_parameters = None, random_basis_layers = None):

        qcirc = QuantumCircuit(self.number_of_qubits)

        if basis == 'x':
            qcirc.h(range(self.number_of_qubits))

        elif basis == 'y':
            qcirc.rz(-np.pi/2, range(self.number_of_qubits))
            qcirc.h(range(self.number_of_qubits))

        elif basis == 'random':
            self.add_random_measurement(meas_parameters, qcirc, random_basis_layers)

        elif basis == 'random_pauli':
            for qubit in range(self.number_of_qubits):
                choice = np.random.choice(['x', 'y', 'z'])
                if choice == 'x':
                    qcirc.h(qubit)

                elif choice == 'y':
                    qcirc.rz(-np.pi/2, qubit)
                    qcirc.h(qubit)

        return qcirc
    
    
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
