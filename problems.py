import numpy as np
import networkx as nx
import collections
from qiskit_optimization.applications import Maxcut

class MaxCut():
    def __init__(self, size, weighted=False, regularity=None, seed=0, edges=None):

        
        if regularity:
            self.graph = nx.random_regular_graph(regularity, size, seed)
            self.edges = len(self.graph.edges)
        
        else:
            self.edges = edges
            self.graph = nx.gnm_random_graph(size, edges, seed=seed)

        self.w = nx.to_numpy_matrix(self.graph, nodelist=sorted(self.graph.nodes()))

        if weighted:
            for i in range(size):
                for j in range(size):
                    if i<j and self.w[i,j] != 0:
                        self.w[i,j] = round(np.random.uniform(0,1), 2)
                        self.w[j,i] = self.w[i,j]
                        
    def get_qubit_operator(self):
        
        maxcut = Maxcut(self.w)
        qp = maxcut.to_quadratic_program()
        qubitOp, offset = qp.to_ising()
        
        return qubitOp, offset

    def optimal_cost_brute_force(self):
        optimal_cost = 0
        number_of_qubits = len(self.w)
        best_string = 0
        costs = collections.defaultdict(list)
        for b in range(2**number_of_qubits):
            x = [int(t) for t in reversed(list(bin(b)[2:].zfill(number_of_qubits)))]
            cost = 0
            for i in range(number_of_qubits):
                for j in range(number_of_qubits):
                    cost += self.w[i,j] * x[i] * (1-x[j])

            cost = np.round(cost,5)
            x.reverse()
            costs[cost].append(x)

            if optimal_cost < cost:
                optimal_cost = cost


        costs = sorted(costs.items())
        optimal_strings = costs[-1][1]
        return optimal_cost, optimal_strings
    
