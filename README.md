# Random Natural Gradient

Qiskit implementation of the paper "Random Natural Gradient" (arXiv:2311.04135).
Authors: Ioannis Kolotouros and Petros Wallden

## Description

This GitHub repository contains a Python implementation of the two main algorithms described in the manuscript. 

The first algorithm is called "Random Natural Gradient" (RNG) and requires the construction of the random classical Fisher information matrix at every iteration. This matrix then transforms the gradient vector, providing a new direction in the parameterized space that takes information about what is happening locally in the space of quantum states. This method can be considered an approximation to the Quantum Natural Gradient that requires quadratically less quantum resources.

The second algorithm is called "Stochastic-Coordinate Quantum Natural Gradient" (SC-QNG). In this algorithm, at every iteration, only a portion of the total parameters are considered and the reduced quantum Fisher information matrix is constructed for this subset. The main intuition behind this algorithm is that only an (unknown) subset of the total parameters can result in an independent change of the underlying quantum state. As such, using all possible resources results in an extra overhead in the classical optimization.


## Installation
pip install -r requirements.txt

## Instructions

maxcut_optimization.py -- This is the main file. You can choose the type of graph (e.g. regular) and different optimization algorithms.

optimizers.py -- This file contains the different classical optimization algorithms used in the paper (QNG, SC-QNG, RNG, and GD).

fisher_matrices.py -- This file constructs the different classical Fisher information matrices and the quantum Fisher information matrix.

problems.py -- This file generates a given MaxCut instance.
