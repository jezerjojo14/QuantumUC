from QAOA_ansatz import create_QAOA_ansatz, create_hhl_circ
from amp_est import real_amp_est
from taylor_precomputation import construct_asin_x_inv_circuit, construct_exp_k_abs_cos_circuit, get_asin_x_inv_expression, get_cos_expression
from math import asin
import cmath
from problem_formulation import Node, Line, Grid, UCProblem

import numpy as np
from scipy import linalg
from math import pi
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library.data_preparation.state_preparation import StatePreparation

from qiskit.quantum_info import Statevector

from qiskit.opflow import PauliTrotterEvolution
from qiskit.opflow import I, X, Y, Z
import os
import importlib
import matplotlib
importlib.reload(matplotlib)
matplotlib.use('Agg')
from matplotlib import pyplot as plt

current_dir=os.getcwd()
final_dir=os.path.join(current_dir, "asin_m_dependence")
os.makedirs(final_dir, exist_ok=True)

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
    expected_plot=np.array([asin_x_inv(x_scale*i/2**n) for i in range(2**n)])
    plt.plot([i/2**n for i in range(2**n)], expected_plot, label="Actual")
    plt.show()
    # bounds=(0,expected_plot[np.isfinite(expected_plot)].max()*1.3)
    # print(bounds)
    # plt.ylim(bounds)
    # plt.savefig(os.path.join(final_dir, "plot_arcsin_x_inv__m_"+str(m)+".png"))
    # plt.clf()
    # plt.close('all')

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


def asin_circ_test(m=5.536580568672519):
    asin_circ=construct_asin_x_inv_circuit(5, 4, 1, m)
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

def get_subsyt_statevectors(sv, target_qubits):
    # sv=list(np.around(np.asarray(sv),3))
    sv=np.asarray(sv)
    n=int(np.log2(len(sv)))
    remaining_qubits=[i for i in range(n) if i not in target_qubits]
    sv_list=[]
    for i in range(2**len(remaining_qubits)):
        sv_el=[]
        active_remaining_qubits=[int(c) for c in str(bin(i))[2:][::-1]]
        index_=sum([on*2**q_num for on,q_num in zip(active_remaining_qubits,remaining_qubits)])
        for j in range(2**len(target_qubits)):
            active_target_qubits=[int(c) for c in str(bin(j))[2:][::-1]]
            index=index_+sum([on*2**q_num for on,q_num in zip(active_target_qubits,target_qubits)])
            sv_el+=[sv[index]]
        sv_el_norm=(sum([np.abs(el)**2 for el in sv_el]))**0.5
        sv_el=np.array(sv_el)
        first_el_phase=cmath.phase(sv_el[0])
        sv_el=sv_el*cmath.exp(-first_el_phase*1j)
        if sv_el_norm:
            sv_el=sv_el/sv_el_norm
        sv_list+=[[[i],sv_el,sv_el_norm**2]]

    return sv_list

def disp_subsyst_statevector(sv,target_qubits):
    sv_list=get_subsyt_statevectors(sv, target_qubits)
    n=int(np.log2(len(sv)))
    no_change=True
    while no_change:
        no_el=len(sv_list)
        for i in range(no_el):
            i=no_el-1-i
            sv=sv_list[i]
            if sv[2]<0.003:
                sv_list.pop(i)
                no_change=False
                continue
            for j in range(i):
                if np.allclose(sv_list[j][1],sv[1]):
                    sv_list[j][0]+=sv[0]
                    sv_list[j][0]=list(set(sv_list[j][0]))
                    sv_list[j][2]+=sv[2]
                    sv_list.pop(i)
                    no_change=False
                    break
    sv_list.sort(key=lambda sv:sv[2])
    for sv in sv_list:
        print("When remaining qubits are in states", ",".join(["|"+(str(bin(ind))[2:]).zfill(n-len(target_qubits))+">" for ind in sv[0]]), " (probability=",sv[2],"),")
        print("the statevector of qubits", ",".join([str(q) for q in target_qubits]), "is")
        print(np.around(sv[1],3))
            
    

def hhl_test_deep():
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
    real_powers=real_powers[:,0]
    B=grid.A
    max_eigval=grid.A_eig_bounds[1]
    min_eigval=grid.A_eig_bounds[0]
    num_time_slices=5
 
    real_powers_norm=np.linalg.norm(np.array(real_powers))
 
    print("Actual solution:\n", linalg.inv(B) @ real_powers)
    print("Predicted simulated sol_n:\n", (min_eigval/real_powers_norm)*(linalg.inv(B) @ real_powers))
 
    # Run code from create_hhl_circ

    print("Min_eigval:",min_eigval)
    print("Max_eigval:",max_eigval)
    
    J, P = linalg.eig(B)

    print("Actual eigvals:",J)

    st0 = Statevector.from_instruction(hhl_circ)
    print("ST0")
    disp_subsyst_statevector(st0,[0,1])
    print()
    print()
    print()
    print()

    print("Constructing HHL Circuit")

    # Rescaling and resizing

    # Normalize the real_powers vector. This is the state we often call |b> when explaining HHL
    real_powers=np.array(real_powers)/real_powers_norm

    # Number of extra dimensions we need to make our vector length a power of 2
    extra_dim=2**len(tot_nodes)-len(real_powers)

    # Make real_powers look more like a statevector
    real_powers=np.concatenate((real_powers, np.array([0 for _ in range(extra_dim)])))
    B=np.array(B)

    # Scale B such that the eigenvalues are scaled in such a way that the largest bitstring (111...111)
    # in the phase register represents the max of 1/C and max_eigval
    
    B=B*2*pi/max_eigval

    B=np.concatenate((B, np.zeros((extra_dim, len(list(B))))))
    B=np.concatenate((B, np.zeros((len(list(B)), extra_dim))), axis=1)
    for i in range(1,extra_dim+1):
        B[-i][-i]=1

    print("Final real_powers:", real_powers)

    # State prep

    hhl_circ.append(StatePreparation(real_powers), tot_nodes)
    # hhl_circ.h(tot_nodes)
    hhl_circ.x(gen_nodes)

    for i in range(len(gen_nodes)):
        # Take the i-th entry in tot_nodes' statevector to the last position in the statevector
        for j in range(len(tot_nodes)):
            if i%(2**(j+1))<2**j:
                hhl_circ.x(tot_nodes[j])
        # Last position in tot_nodes' statevector corresponds to the basis state where all qubits in tot_nodes are 1
        # The following statement flips the ancilla when all tot_nodes's qubits are 1 and when gen_nodes[i] is one.
        # The first half of the combined statevector of tot_nodes and state_prep_anc is the exact statevector
        # that we're looking for.
        hhl_circ.mcx([gen_nodes[i]]+list(tot_nodes),state_prep_anc[0])
        # Undo the shuffling we did earlier.
        for j in range(len(tot_nodes)):
            if i%(2**(j+1))<2**j:
                hhl_circ.x(tot_nodes[j])

    hhl_circ.x(gen_nodes)

    st1 = Statevector.from_instruction(hhl_circ)
    print("ST1:")
    disp_subsyst_statevector(st1,[2,3,4])
    print()
    print()
    print()
    print()

    # Paulinomial decomposition

    H=I
    for i in range(len(tot_nodes)-1):
        H=H^I
    
    print("B:\n",B)
    print()

    H=0*H

    paulis=[I,X,Y,Z]

    for i in range(4**len(tot_nodes)):
        term=paulis[i%4]
        i=i//4
        for j in range(len(tot_nodes)-1):
            term=term^paulis[i%4]
            i=i//4
        a=sum([((term.to_matrix())@(B))[k][k] for k in range(2**len(tot_nodes))])/(2**len(tot_nodes))
        H+=(a*term)

    # Create evolution circuit


    # exp_i(A) := exp(-iA)
    evolution_op = (-H).exp_i()

    trotterized_op = PauliTrotterEvolution(
                    trotter_mode='trotter',
                    reps=num_time_slices).convert(evolution_op)
    
    trot_exp_H=trotterized_op.to_matrix()

    J, P = linalg.eig(trot_exp_H)
    J=[np.log((el**(-1j)).real) for el in J]
    J=[el+2*np.pi*int(el<0) for el in J]
    print("Eigenvalues of matrix:",J)
    
    L=np.array([[int(i==j)*J[i] for j in range(len(J))] for i in range(len(J))])
    H_rev_eng = P @ L @ linalg.inv(P)
    print("The hamiltonian actually being simulated\nby the approximated evolution operator:\n", H_rev_eng)
    print()

    U=trotterized_op.to_circuit().to_gate()
    CU=U.control(2)

    # QPE

    hhl_circ_temp=QuantumCircuit(gen_nodes,tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc, name="hhl_circ_temp")
    hhl_circ_temp.h(hhl_phase_reg)
    hhl_circ_temp.x(state_prep_anc)
    repetitions = 1
    for counting_qubit in range(len(hhl_phase_reg)):
        for i in range(repetitions):
            # Applying rotation to phase register qubits in reverse order
            # This order will be switched back after applying inverse qft
            hhl_circ_temp.append(CU, [hhl_phase_reg[len(hhl_phase_reg)-1-counting_qubit]]+[state_prep_anc[0]]+[q for q in tot_nodes])
        repetitions *= 2
    
    st2 = Statevector.from_instruction(hhl_circ)
    print("ST2:")
    disp_subsyst_statevector(st2,[2,3,4])
    print()
    print()
    print()
    print()


    def qft_dagger(n):
        """n-qubit QFTdagger the first n qubits of hhl_phase_reg"""
        # DO forget the Swaps!
        # for qubit in range(n//2):
        #     qc.swap(qubit, n-qubit-1)
        for j in range(n):
            for m in range(j):
                hhl_circ_temp.cp(-np.pi/float(2**(j-m)), hhl_phase_reg[m], hhl_phase_reg[j])
            hhl_circ_temp.h(hhl_phase_reg[j])

    qft_dagger(len(hhl_phase_reg))

    # print(hhl_circ_temp.draw())

    hhl_circ.compose(hhl_circ_temp, inplace=True)

    
    st3 = Statevector.from_instruction(hhl_circ)
    print("ST3:")
    disp_subsyst_statevector(st3,[2,3,4])
    print()
    print()
    print()
    print()

    # Conditioned rotations

    hhl_circ.compose(construct_asin_x_inv_circuit(len(hhl_phase_reg),4,1,max_eigval/min_eigval), [q for q in hhl_phase_reg]+[hhl_anc[0]], inplace=True)
    hhl_circ.x(hhl_anc[0])
    # disp_subsyst_statevector(st4,[2,3])

    # Uncompute QPE to unentangle the ancillas

    hhl_circ.compose(hhl_circ_temp.inverse(), inplace=True)

    # print(hhl_circ.draw())

    st4 = Statevector.from_instruction(hhl_circ)
    print("ST4:")
    disp_subsyst_statevector(st4,[2,3])
    print()
    print()
    print()
    print()



    
    # Run HHL circuit and get statevector
    
    backend=Aer.get_backend('aer_simulator')
    hhl_circ.save_statevector()
    hhl_circ=transpile(hhl_circ, backend)
    result = backend.run(hhl_circ).result()
    statevector = result.get_statevector(hhl_circ)

    # Print relevant part of statevector

    print(list(np.around(np.asarray(statevector),3))[:32])



# Run tests

# Success
# for m in range(70):
#     m+=2
#     plot_asin_x_inv(5,4,m)

#     statevector=asin_circ_test(m)

#     plt.plot([i/32 for i in range(32)], [abs(el) for el in list(statevector)[32:]])
#     expected_plot=np.array([1/(m*(x/32)*2**(5/2)) if x!=0 else np.nan for x in range(32)])
#     plt.plot([i/32 for i in range(32)], expected_plot)
#     # plt.plot([i/32 for i in range(32)], [1/((x/32)*2**(5/2)) if x!=0 else np.nan for x in range(32)])
#     # plt.show()
#     bounds=(0,expected_plot[np.isfinite(expected_plot)].max()*1.3)
#     print(bounds)
#     plt.ylim(bounds)
#     # plt.ylim((-1,max(expected_plot)))
#     plt.savefig(os.path.join(final_dir, "plot_x_inv__m_"+str(m)+".png"))
#     plt.clf()
#     plt.close('all')


# asin_circ_test()
# Success


# hhl_test()
# Test failed. Answer is coming out as
# [(0.685917997575626-1.2319469437860386e-13j), (0.3429589987878453-5.18633872628292e-14j), (-0.3429589987878187+6.504051088580612e-14j), (-0.5144384981817477+7.863900559078633e-14j)]
# Approx
# Actual answer is array([ 0.5  ,  0.375, -0.75 , -0.875])
# [(0.03901843403928501-5.90736734281494e-15j), (0.019509217019644462-2.360912436517677e-15j), (-0.019509217019642477+3.1570750431305193e-15j), (-0.029263825529466245+3.910317383165953e-15j)]

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

hhl_test_deep()