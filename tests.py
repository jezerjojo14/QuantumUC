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
        active_bits=[n-1-m for m in range(4) if (format(i, 'b').zfill(n))[m]=='1']
        print(active_bits)
        for j in range(len(asin_x_inv_expression)):
            term=[asin_x_inv_expression[j], [n-1-m for m in range(n) if (format(j, 'b').zfill(n))[m]=='1']]
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
    # print()
    print(list(np.around(np.asarray(statevector),3)))


def asin_circ_test():
    asin_circ=construct_asin_x_inv_circuit(5, 4, 2, 5.536580568672519)
    input_reg=QuantumRegister(5)
    ry_anc=QuantumRegister(1)
    qc=QuantumCircuit(input_reg, ry_anc)
    qc.h(input_reg)
    qc.compose(asin_circ, inplace=True)
    backend=Aer.get_backend('aer_simulator')
    qc.save_statevector()
    qc=transpile(qc, backend)
    result = backend.run(qc).result()
    statevector = result.get_statevector(qc)
    return statevector




# def hhl_state_prep(hhl_circ):
#     hhl_circ=QuantumCircuit(gen_nodes,tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc, name="hhl_circ")


# Run tests

# plot_asin_x_inv(5,4,5.536580568672519)
hhl_test()
# Test failed. Answer is coming out as
# [(0.685917997575626-1.2319469437860386e-13j), (0.3429589987878453-5.18633872628292e-14j), (-0.3429589987878187+6.504051088580612e-14j), (-0.5144384981817477+7.863900559078633e-14j)]
# Approx
# Actual answer is array([ 0.5  ,  0.375, -0.75 , -0.875])
# [(0.03901843403928501-5.90736734281494e-15j), (0.019509217019644462-2.360912436517677e-15j), (-0.019509217019642477+3.1570750431305193e-15j), (-0.029263825529466245+3.910317383165953e-15j)]

# statevector=asin_circ_test()

# plt.plot(range(32), [abs(el) for el in list(statevector)[32:]])
# plt.plot(range(32), [1/(x) if x!=0 else np.nan for x in range(32)])
# plt.show()

# [[ 3. -1.  0. -1.]
#  [-1.  2. -1.  0.]
#  [ 0. -1.  2. -1.]
#  [-1.  0. -1.  2.]]

# [ 2.   1.  -1.  -1.5]



# [[-3.40454829+0.j  1.13484943+0.j  0.        +0.j  1.13484943+0.j]
#  [ 1.13484943+0.j -2.26969886+0.j  1.13484943+0.j  0.        +0.j]
#  [ 0.        +0.j  1.13484943+0.j -2.26969886+0.j  1.13484943+0.j]
#  [ 1.13484943+0.j  0.        +0.j  1.13484943+0.j -2.26969886+0.j]]


# [[-0.16342566-0.27164328j  0.01833781+0.37063308j  0.70230126-0.36175213j
#    0.01833781+0.37063308j]
#  [ 0.01833781+0.37063308j -0.04308225+0.2513932j   0.36948844+0.18975702j
#    0.60029566-0.51415553j]
#  [ 0.70230126-0.36175213j  0.36948844+0.18975702j -0.14508785+0.09898981j
#    0.36948844+0.18975702j]
#  [ 0.01833781+0.37063308j  0.60029566-0.51415553j  0.36948844+0.18975702j
#   -0.04308225+0.2513932j ]]

# [[-0.16570573-0.26528365j, -0.02045629+0.37106175j,  0.7047277 -0.35960324j, 0.02169489+0.3709914j, ] [ 0.06356696+0.36614823j, -0.04409058+0.24854458j,  0.3865175 +0.14947224j, 0.53688608-0.58154272j,] [ 0.7047277 -0.35960324j,  0.34296922+0.23261523j, -0.1444304 +0.09853308j,   0.36710477+0.19228058j,] [ 0.02169489+0.3709914j,   0.65397874-0.44581432j,  0.36710477+0.19228058j,  -0.04409058+0.24854458j,]]