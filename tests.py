from QAOA_ansatz import create_QAOA_ansatz, create_hhl_circ
from amp_est import real_amp_est
from taylor_precomputation import construct_asin_x_inv_circuit, construct_exp_k_abs_cos_circuit, get_asin_x_inv_expression, get_cos_expression
from math import asin
from problem_formulation import Node, Line, Grid, UCProblem

import numpy as np
from math import pi
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library.data_preparation.state_preparation import StatePreparation

from qiskit.opflow import PauliTrotterEvolution
from qiskit.opflow import I, X, Y, Z

from matplotlib import pyplot as plt

def asin_x_inv(x):
    try:
        return asin(1/x)
    except:
        return np.nan

def plot_asin_x_inv(n, no_terms, x_scale):
    asin_x_inv_expression=get_asin_x_inv_expression(n, no_terms, x_scale)
    y=[0 for _ in range(2**n)]
    for i in range(len(y)):
        active_bits=[4-1-m for m in range(4) if (format(i, 'b').zfill(4))[m]=='1']
        print(active_bits)
        for j in range(len(asin_x_inv_expression)):
            term=[asin_x_inv_expression[j], [4-1-m for m in range(4) if (format(j, 'b').zfill(4))[m]=='1']]
            apply_rotation=True
            for bit in term[1]:
                if bit not in active_bits:
                    apply_rotation=False
            if apply_rotation:
                y[i]+=term[0]

    plt.plot([i/2**n for i in range(2**n)],y, label="Taylor approx")
    plt.plot([i/2**n for i in range(2**n)], [asin_x_inv(x_scale*i/2**n) for i in range(2**n)], label="Actual")
    plt.show()


plot_asin_x_inv(4,4,5.721133254294936)