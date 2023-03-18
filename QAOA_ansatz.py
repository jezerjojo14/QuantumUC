import numpy as np
from scipy import linalg

from math import pi
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library.data_preparation.state_preparation import StatePreparation

from qiskit.opflow import PauliTrotterEvolution
from qiskit.opflow import I, X, Y, Z

from amp_est import real_amp_est
from taylor_precomputation import construct_asin_x_inv_circuit, construct_exp_k_abs_cos_circuit

from qiskit.circuit import qpy_serialization

import json
import os

import time

import pickle

def general_CZ(F, n):
    x=QuantumRegister(n, 'x')
    y=QuantumRegister(n, 'y')
    qc_CZ=QuantumCircuit(x,y)
    for k in range(n):
        for i in range(n):
            j=(i+k)%n
            qc_CZ.cp(2*pi*2**(i+j-n)/F,x[i],y[j])
    return qc_CZ

# Function copied from qiskit textbook that implements a qft without the swaps
def qft_rotations(circuit, n):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(circuit, n)

def create_hhl_circ(real_powers,B,max_eigval,C,gen_nodes,tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc,num_time_slices=3):

    """
    Creates a quantum circuit to perform HHL.

    real_powers (list | numpy.array): Contains real powers of all nodes
    B (list of lists | numpy.array with dimension 2): Susceptance matrix of all nodes
    max_eigval (float): Upper bound for maximum eigenvalue of B
    C (float): Lower bound for minimum eigenvalue of B
    gen_nodes (QuantumRegister): Contains qubits each representing a generator node
    tot_nodes (QuantumRegister): Contains qubits where each basis state represents a node
    stat_prep_anc (QuantumRegister): Contains one qubit used to prepare the initial state
    hhl_phase_reg (QuantumRegister): Register of qubits for storing eigenvalues during HHL
    hhl_anc (QuantumRegister): Contains one qubit used for eigenvalue inversion during HHL
    num_time_slices (int): Number of time slices used for trotterized hamiltonian evolution

    Returns:
    hhl_circ (QuantumCircuit): Quantum circuit that performs HHL

    """

    hhl_circ=QuantumCircuit(gen_nodes,tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc)

    with open("circuit_ID.json") as f:
        circuit_IDs=json.load(f)

    current_dir=os.getcwd()
    circuits_dir=os.path.join(current_dir, "circuits")
    
    circuit_key="HHL_"+str(real_powers)+"_"+str(B)+"_"+str(max_eigval)+"_"+str(C)+\
                "_"+str(len(hhl_phase_reg))+"_"+str(num_time_slices)

    print("Check if HHL circuit already exists")

    circuit_ID=False

    try:
        circuit_ID=circuit_IDs[circuit_key]
        print("Circuit exists. Circuit ID:", circuit_ID)
    except Exception as e:
        print(e)

    if circuit_ID:
        with open(os.path.join(circuits_dir,circuit_ID+'.qpy'), 'rb') as fd:
            circuit = qpy_serialization.load(fd)[0]
            hhl_circ.compose(circuit, inplace=True)
        return hhl_circ
    
    
    print("Constructing HHL Circuit")

    # Rescaling and resizing

    real_powers_norm=np.linalg.norm(np.array(real_powers))
    real_powers=np.array(real_powers)/real_powers_norm

    extra_dim=2**len(tot_nodes)-len(real_powers)

    real_powers=np.concatenate((real_powers, np.array([0 for _ in range(extra_dim)])))
    B=np.array(B)
    print("Initial B:", B)
    B=B*2*pi*(2**(len(hhl_phase_reg)-1))/(max_eigval*(2**len(hhl_phase_reg)))

    B=np.concatenate((B, np.zeros((extra_dim, len(list(B))))))
    B=np.concatenate((B, np.zeros((len(list(B)), extra_dim))), axis=1)
    for i in range(1,extra_dim+1):
        B[-i][-i]=1

    print("Final B:", B)

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

    # Paulinomial decomposition

    H=I
    for i in range(len(tot_nodes)-1):
        H=H^I
    
    H=0*H

    paulis=[I,X,Y,Z]

    for i in range(4**len(tot_nodes)):
        term=paulis[i%4]
        i=i//4
        for j in range(len(tot_nodes)-1):
            term=term^paulis[i%4]
            i=i//4
        a=sum([((term.to_matrix())@(-B))[k][k] for k in range(2**len(tot_nodes))])/(2**len(tot_nodes))
        H+=(a*term)

    # Create evolution circuit

    evolution_op = (H).exp_i()

    trotterized_op = PauliTrotterEvolution(
                    trotter_mode='trotter',
                    reps=num_time_slices).convert(evolution_op)
    U=trotterized_op.to_circuit().to_gate()
    CU=U.control(2)

    # QPE

    hhl_circ_temp=QuantumCircuit(gen_nodes,tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc)
    hhl_circ_temp.h(hhl_phase_reg)
    hhl_circ_temp.x(state_prep_anc)
    repetitions = 1
    for counting_qubit in range(len(hhl_phase_reg)):
        for i in range(repetitions):
            hhl_circ_temp.append(CU, [hhl_phase_reg[len(hhl_phase_reg)-1-counting_qubit]]+[state_prep_anc[0]]+[q for q in tot_nodes])
        repetitions *= 2

    # hhl_circ_temp.x(state_prep_anc)

    def qft_dagger(n):
        """n-qubit QFTdagger the first n qubits in circ"""
        # DO forget the Swaps!
        # for qubit in range(n//2):
        #     qc.swap(qubit, n-qubit-1)
        for j in range(n):
            for m in range(j):
                hhl_circ_temp.cp(-pi/float(2**(j-m)), hhl_phase_reg[m], hhl_phase_reg[j])
            hhl_circ_temp.h(j)

    qft_dagger(len(hhl_phase_reg))

    hhl_circ.compose(hhl_circ_temp, inplace=True)

    # Conditioned rotations

    hhl_circ.compose(construct_asin_x_inv_circuit(len(hhl_phase_reg),4,C,max_eigval), [q for q in hhl_phase_reg]+[hhl_anc[0]], inplace=True)

    # Uncompute QPE to unentangle the ancillas

    hhl_circ.compose(hhl_circ_temp.inverse(), inplace=True)

    circuit_ID=str(len(os.listdir(circuits_dir)))

    circuit_IDs[circuit_key]=circuit_ID
    
    with open("circuit_ID.json", 'w', encoding='utf-8') as f:
        json.dump(circuit_IDs, f, ensure_ascii=False, indent=4)
    

    print("Length of untranspiled HHL", len(hhl_circ))


    print("Transpiling HHL circuit for storage")
    simulator = Aer.get_backend('aer_simulator')
    hhl_circ=transpile(hhl_circ, simulator, optimization_level=3)
    
    with open(os.path.join(circuits_dir,circuit_ID+'.qpy'), 'wb') as fd:
        print("Length of transpiled HHL", len(hhl_circ))
        qpy_serialization.dump(hhl_circ, fd)

    return(hhl_circ)












def create_QAOA_ansatz(
    timestep_count, gen_node_count, real_powers, hhl_phase_qubit_count, qadc_qubit_count,
    running_costs, on_off_costs, line_costs, B, max_eigval, C,
    no_layers, consider_transmission_costs=True
    # gamma_values, beta_values
    ):

    """
    Creates a parametrized QAOA ansatz circuit.

    Parameters:
    timestep_count (int): Number of timesteps in our UC problem
    gen_node_count (int): Number of generator nodes
    real_powers (2d numpy.array | list of lists): Real powers with time of all nodes
    hhl_phase_qubit_count (int): Number of qubits to store eigenvalues in HHL procedure
    qadc_qubit_count (int): Number of qubits to store output of amplitude estimation
    running_costs (numpy.array | list): Costs of keeping a generator node running for a timestep
    on_off_costs (list with length 2, each element is a list or array): on_off_costs[0] contains costs of turning a generator node on
                                                                        on_off_costs[1] contains costs of turning a generator node off
    line_costs (list of lists | numpy.array with dimension 2): line_costs[i][j] contains the cost of transmission per unit power
                                                               per unit time of the line connecting nodes i and j
    B (list of lists | numpy.array with dimension 2): Susceptance matrix of all nodes
    max_eigval (float): Upper bound for maximum eigenvalue of B
    C (float): Lower bound for minimum eigenvalue of B
    no_layers (int): Number of layers in QAOA

    Returns:
    qc (QuantumCircuit): Parametrized circuit for QAOA

    """

    # if len(beta_values)!=len(gamma_values):
    #     raise
    # no_layers=len(gamma_values)

    with open("circuit_ID.json") as f:
        circuit_IDs=json.load(f)

    current_dir=os.getcwd()
    circuits_dir=os.path.join(current_dir, "circuits")

    QAOA_circuit_key="QAOA_"+"_"+str(timestep_count)+"_"+str(gen_node_count)+"_"+str(real_powers)+\
            "_"+str(hhl_phase_qubit_count)+"_"+str(qadc_qubit_count)+"_"+str(running_costs)+\
            "_"+str(on_off_costs)+"_"+str(line_costs)+"_"+str(B)+"_"+str(max_eigval)+\
            "_"+str(C)+"_"+str(consider_transmission_costs)
    
    circuit_ID=False

    try:
        print("Check if QAOA circuit exists")
        circuit_ID=circuit_IDs[QAOA_circuit_key]
        print("Circuit exists. Circuit ID:", circuit_ID)
    except Exception as e:
        print(e)
    
    params=ParameterVector('p', 2*no_layers)
    
    node_count=len(real_powers)

    gen_nodes=[QuantumRegister(gen_node_count, "gen_nodes_at_t_"+str(i)) for i in range(timestep_count)]
    tot_nodes=QuantumRegister(int(np.ceil(np.log2(node_count))), "tot_nodes")
    state_prep_anc=QuantumRegister(1, "state_prep_anc")
    hhl_phase_reg=QuantumRegister(hhl_phase_qubit_count, "hhl_phase_reg")
    hhl_anc=QuantumRegister(1, "hhl_anc")
    qadc_reg=QuantumRegister(qadc_qubit_count, "qadc_reg")
    qadc_anc=QuantumRegister(1, "qadc_anc")

    output_reg=[ClassicalRegister(gen_node_count) for i in range(timestep_count)]

    qc_total=QuantumCircuit(*gen_nodes,tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc,qadc_reg,qadc_anc,*output_reg)

    
    if circuit_ID:
        with open(os.path.join(circuits_dir,circuit_ID+'.qpy'), 'rb') as fd:
            qc = qpy_serialization.load(fd)[0]
    else:    
        circuit_ID=False

        print("Constructing QAOA Circuit")
        qc=QuantumCircuit(*gen_nodes,tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc,qadc_reg,qadc_anc,*output_reg)

        print("Total number of qubits in our circuit:", qc.num_qubits)

        # Use HHL Phase reg for penalty adder
        qft=QuantumCircuit(hhl_phase_qubit_count)
        qft_rotations(qft, hhl_phase_qubit_count)

        for gen_nodes_reg in gen_nodes:
            qc.h(gen_nodes_reg)

        # First element gamma, second beta
        params_temp=ParameterVector('p_temp', 2)

        # Cost layer

        # Running costs
        # print("layer_index", layer_index)
        # print("params", params)
        for t in range(timestep_count):
            for i in range(gen_node_count):
                qc.rz(params_temp[0]*running_costs[i], gen_nodes[t][i])

        # On off costs
        for t in range(timestep_count-1):
            for i in range(gen_node_count):
                # On costs
                qc.x(gen_nodes[t][i])
                qc.crz(params_temp[0]*on_off_costs[0][i], gen_nodes[t][i], gen_nodes[t+1][i])
                qc.x(gen_nodes[t][i])

                # Off costs
                qc.x(gen_nodes[t+1][i])
                qc.crz(params_temp[0]*on_off_costs[1][i], gen_nodes[t][i], gen_nodes[t+1][i])
                qc.x(gen_nodes[t+1][i])

        # Transmission Costs

        if consider_transmission_costs:
            for t in range(timestep_count):
                hhl_circ=create_hhl_circ([r[t] for r in real_powers],B,max_eigval,C,gen_nodes[0],tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc)
                for i in range(node_count):
                    for j in range(i):
                        C_L=line_costs[i][j]

                        if C_L:

                            exp_k_abs_cos_circuit=construct_exp_k_abs_cos_circuit(qadc_qubit_count,4,abs(B[i][j])*C_L*np.linalg.norm(np.array(real_powers))*params_temp[0])

                            # Here we set the 0-th component of the statevector at the end of the hhl circuit to theta_i-theta_j

                                # Move theta_i to 0th component of statevector, also update j accordingly to track the position of theta_j
                            for k in range(len(tot_nodes)):
                                if i%(2**(k+1))>=2**k:
                                    hhl_circ.x(tot_nodes[k])
                                    if j%(2**(k+1))>=2**k:
                                        j-=2**k
                                    else:
                                        j+=2**k
                                # Set k to be the position of the most significant 1 in binary expansion of j
                            k=len(tot_nodes)-1
                            while True:
                                if j<2**k:
                                    k-=1
                                    continue
                                break
                                # Move theta_j to 2**k-th component
                            for l in range(k):
                                    # If l-th digit of binary expansion of j is 1
                                if j%(2**(l+1))>=2**l:
                                    hhl_circ.cx(tot_nodes[k],tot_nodes[l])
                            hhl_circ.h(k)
                            hhl_circ.x(k)

                            with open("circuit_ID.json") as f:
                                circuit_IDs=json.load(f)

                            current_dir=os.getcwd()
                            circuits_dir=os.path.join(current_dir, "circuits")
                            
                            circuit_key="QADC_"+str([r[t] for r in real_powers])+"_"+str(B)+"_"+str(max_eigval)+"_"+str(C)+\
                                        "_"+str(len(hhl_phase_reg))+"_"+str(qadc_qubit_count)+"_"+str((i,j))

                            print("Check if QADC circuit already exists.")

                            try:
                                circuit_ID=circuit_IDs[circuit_key]
                                print("Exists. Circuit ID is", circuit_ID)
                            except Exception as e:
                                print(e)
                                print("Constructing QADC Circuit")
                                qadc_circ=real_amp_est(gen_node_count+len(tot_nodes)+2+hhl_phase_qubit_count,0,hhl_circ,qadc_qubit_count)
                                circuit_ID=str(len(os.listdir(circuits_dir)))

                                circuit_IDs[circuit_key]=circuit_ID
                                
                                with open("circuit_ID.json", 'w', encoding='utf-8') as f:
                                    json.dump(circuit_IDs, f, ensure_ascii=False, indent=4)
                                
                                with open(os.path.join(circuits_dir,circuit_ID+'.qpy'), 'wb') as fd:
                                    qpy_serialization.dump(qadc_circ, fd)
                            if circuit_ID:
                                with open(os.path.join(circuits_dir,circuit_ID+'.qpy'), 'rb') as fd:
                                    qadc_circ = qpy_serialization.load(fd)[0]
                            qc.compose(qadc_circ, [q for q in qadc_reg]+[q for q in gen_nodes[t]]+[q for q in tot_nodes]+[state_prep_anc[0]]+[q for q in hhl_phase_reg]+[hhl_anc[0]]+[qadc_anc[0]])
                            qc.compose(exp_k_abs_cos_circuit, qadc_reg)
                            qc.compose(qadc_circ.inverse(), [q for q in qadc_reg]+[q for q in gen_nodes[t]]+[q for q in tot_nodes]+[state_prep_anc[0]]+[q for q in hhl_phase_reg]+[hhl_anc[0]]+[qadc_anc[0]])

        # Penalty Costs
        C_P=sum(running_costs)+sum(on_off_costs[0])+sum(on_off_costs[1])
        if consider_transmission_costs:
            C_P+=sum([sum(arr) for arr in line_costs])/2

        for t in range(timestep_count):
            qc.h(hhl_phase_reg)
            for i in range(gen_node_count):
                # a_i is real power in units where |100...0> is equal to the total demand power
                a_i=real_powers[i][t]*2**(hhl_phase_qubit_count-1)/sum([-real_powers[node][t] for node in range(gen_node_count,len(real_powers))])
                for j in range(hhl_phase_qubit_count):
                    qc.cp(2*pi*2**(j-hhl_phase_qubit_count)*a_i,gen_nodes[i],((list(hhl_phase_reg))[::-1])[j])
                # qc.compose(general_CZ(1.0/c_coeff,n), qubits=(list(hhl_phase_reg))[::-1]+list(gen_nodes[t]), inplace=True)
            qc.compose(qft.inverse(), qubits=hhl_phase_reg, inplace=True)
            qc.rz(-params_temp[0]*C_P, hhl_phase_reg[-1])
            qc.compose(qft, qubits=hhl_phase_reg, inplace=True)
            for i in range(gen_node_count):
                a_i=real_powers[i][t]*2**(hhl_phase_qubit_count-1)/sum([-real_powers[node][t] for node in range(gen_node_count,len(real_powers))])
                for j in range(hhl_phase_qubit_count):
                    qc.cp(-2*pi*2**(j-hhl_phase_qubit_count)*a_i,gen_nodes[i],((list(hhl_phase_reg))[::-1])[j])
            qc.h(hhl_phase_reg)

            

        # Mixer layer
        for t in range(timestep_count):
            for i in range(gen_node_count):
                qc.rx(params_temp[1], gen_nodes[t][i])
    

        circuit_ID=str(len(os.listdir(circuits_dir)))

        circuit_IDs[QAOA_circuit_key]=circuit_ID
        
        with open("circuit_ID.json", 'w', encoding='utf-8') as f:
            json.dump(circuit_IDs, f, ensure_ascii=False, indent=4)
        
        with open(os.path.join(circuits_dir,circuit_ID+'.qpy'), 'wb') as fd:
            qpy_serialization.dump(qc, fd)



    for layer_index in range(no_layers):
        qc.assign_parameters([params[layer_index], params[no_layers+layer_index]], inplace=True)
        print("Parameters of qc: ", qc.parameters)
        qc_total.compose(qc,inplace=True)

    for t in range(timestep_count):
        qc_total.measure(gen_nodes[t], output_reg[t])
    
    print("Parameters of qc_total: ", qc_total.parameters)



    
    return qc_total