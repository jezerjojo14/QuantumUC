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

def hhl_test():
    # Create problem
    node1=Node([2,2], 5, 1, 1, "gen1")
    node2=Node([1,1], 1, 2, 1, "gen2")
    node3=Node([-1.5,-1], 0,0,0, "load1")
    node4=Node([-1,0], 0,0,0, "load2")
    line1=Line(node1,node3,1,1)
    line2=Line(node2,node1,1,1)
    line3=Line(node4,node2,1,1)
    line4=Line(node4,node3,1,1)

    problem_instance=UCProblem([line1,line2,line3, line4], [node1,node2,node3,node4], 2)
    grid=problem_instance.grid_timesteps

    real_powers=np.array([node.real_power for node in grid.nodes])

    # Create HHL circuit for the problem

    gen_nodes=QuantumRegister(2,"gen_nodes")
    tot_nodes=QuantumRegister(2,"tot_nodes")
    state_prep_anc=QuantumRegister(1,"state_prep_anc")
    hhl_phase_reg=QuantumRegister(5,"hhl_phase_reg")
    hhl_anc=QuantumRegister(1,"hhl_anc")

    hhl_circ=QuantumCircuit(gen_nodes,tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc)

    hhl_circ.x(gen_nodes)
    hhl_circ.compose(create_hhl_circ(real_powers=real_powers[:,0],
                            B=grid.A,
                            max_eigval=grid.A_eig_bounds[1],
                            C=grid.A_eig_bounds[0],
                            gen_nodes=gen_nodes,
                            tot_nodes=tot_nodes,
                            state_prep_anc=state_prep_anc,
                            hhl_phase_reg=hhl_phase_reg,
                            hhl_anc=hhl_anc, num_time_slices=5), inplace=True)
    
    # Run HHL circuit and get statevector
    
    backend=Aer.get_backend('aer_simulator')
    hhl_circ.save_statevector()
    hhl_circ=transpile(hhl_circ, backend)
    result = backend.run(hhl_circ).result()
    statevector = result.get_statevector(hhl_circ)

    # Print relevant part of statevector

    output=[statevector[3],
    statevector[7],
    statevector[11],
    statevector[15]]
    print(output)


def asin_circ_test():
    asin_circ=construct_asin_x_inv_circuit(5, 4, 1, 5.536580568672519)
    input_reg=QuantumRegister(5)
    ry_anc=QuantumRegister(1)
    qc=QuantumCircuit(input_reg, ry_anc)



def hhl_state_prep(hhl_circ):
    pass

# Run tests

# plot_asin_x_inv(4,4,5.721133254294936)
hhl_test()
# Test failed. Answer is coming out as
# [(0.685917997575626-1.2319469437860386e-13j), (0.3429589987878453-5.18633872628292e-14j), (-0.3429589987878187+6.504051088580612e-14j), (-0.5144384981817477+7.863900559078633e-14j)]
# Approx
# Actual answer is array([ 0.5  ,  0.375, -0.75 , -0.875])