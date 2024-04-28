from qiskit.visualization import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import networkx as nx
from optimizers import Optimizers
from problems import MaxCut

seed = 3

#Problem properties
weighted = True
regularity = 3
number_of_qubits = 6
problem = MaxCut(number_of_qubits, regularity=regularity, weighted=weighted, seed=seed)
optimal_cost, optimal_strings = problem.optimal_cost_brute_force()
qubitOp, offset = problem.get_qubit_operator()


#Properties of the ansatz
layers = 2
maxiter = 500
eta = 0.05
single_qubit_gates = ['ry', 'rx']
entanglement_gates = ['cz']
entanglement = 'linear'
ansatz = 'HEA'
np.random.seed(seed)
random_basis_layers = 2


#There are extra properties on how to calculate the expectation values (i.e. use_shots, number_of_shots) see optimizers.py

print(f'The optimal cost is {optimal_cost} and the strings corresponding to the optimal solution are {optimal_strings}')

 
number_of_parameters = 2*(layers+1)*number_of_qubits


reduced_parameters = int(number_of_parameters/2)
thetas = [np.random.uniform(0, 2*np.pi) for _ in range(number_of_parameters)]


optimizer_statevector = Optimizers(number_of_qubits, layers, thetas, maxiter, ansatz, single_qubit_gates, entanglement_gates, entanglement,
                                                problem = {'type':'MaxCut', 'operator':qubitOp, 'offset':offset})



#Gradient Descent:
#vanilla_gradient_descent_exp_values = optimizer_statevector.gradient_descent(eta=eta)

#Random Natural Gradient:
#random_natural_gradient_exp_values = optimizer_statevector.random_natural_gradient(eta=eta, basis='random', random_basis_layers = random_basis_layers)

#Quantum Natural Gradient:
#quantum_natural_gradient_exp_values = optimizer_statevector.quantum_natural_gradient(eta=eta)

#Stochastic-Coordinate Quantum Natural Gradient
scqng_exp_values = optimizer_statevector.stochastic_quantum_natural_gradient(reduced_parameters, eta=eta)

optimal_exp_values = [-optimal_cost for _ in range(maxiter+1)]


#And below we can plot how the optimizers compare with each other.

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)  


plt.figure(figsize=(10, 7.5))    
ax = plt.subplot(111)

x_axis = range(maxiter+1)

ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()

plt.xticks(fontsize=14)


plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                labelbottom="on", left="off", right="off", labelleft="on")



optimal_exp_values = [-optimal_cost for _ in x_axis]


plt.plot(x_axis, vanilla_gradient_descent_exp_values, lw=2,linestyle='dashdot', color=tableau20[0], label='Vanilla Gradient Descent')
plt.plot(x_axis, random_natural_gradient_exp_values, lw=2,linestyle='dashdot', color=tableau20[2], label='Random Natural Gradient')
plt.plot(x_axis, quantum_natural_gradient_exp_values, lw=2,linestyle='dashdot', color=tableau20[4], label='Quantum Natural Gradient')
plt.plot(x_axis, optimal_exp_values, linestyle='-', color = tableau20[6], label='Optimal Expectation Value')


plt.xlabel('Optimization Iterations', fontsize = 12)
plt.ylabel('Expectation Value', fontsize=12)
plt.legend(fontsize=10, loc='upper right')
plt.show()

