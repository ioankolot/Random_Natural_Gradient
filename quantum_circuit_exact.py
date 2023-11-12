from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from qiskit.circuit.library import TwoLocal
import numpy as np

class Qcir_Exact():

    def __init__(self, number_of_qubits, thetas, layers, ansatz, single_qubit_gates=None, entangled_gates=None, entanglement=None, adjacency_matrix=None, measure_basis = None):
        
        self.number_of_qubits = number_of_qubits
        self.qcir = QuantumCircuit(number_of_qubits)

        if ansatz == 'HEA':
            self.qcir += TwoLocal(num_qubits = number_of_qubits, rotation_blocks = single_qubit_gates, entanglement_blocks = entangled_gates, reps = layers, entanglement=entanglement)

            self.qcir = self.qcir.assign_parameters(thetas)


        elif ansatz == 'HVA':

            self.qcir.h(range(number_of_qubits))
            for layer in range(layers):

                for qubit1 in range(number_of_qubits):
                    for qubit2 in range(number_of_qubits):
                        if qubit1<qubit2:
                            if adjacency_matrix[qubit1, qubit2] != 0:
                                self.qcir.rzz(thetas[2*layer], qubit1, qubit2)

            
                self.qcir.rx(thetas[2*layer+1], range(number_of_qubits))

        if measure_basis == 'X':
            self.qcir.h(range(self.number_of_qubits))


        elif measure_basis == 'Y':
            self.qcir.rz(-np.pi/2, range(self.number_of_qubits))
            self.qcir.h(range(self.number_of_qubits))
        

