import numpy as np
from math import pi
from scipy import linalg
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import Aer
from taylor_precomputation import construct_asin_x_inv_circuit
from qiskit.opflow import PauliTrotterEvolution
from qiskit.opflow import I, X, Y, Z

from qiskit.circuit import qpy_serialization

import json
import os

from qiskit.circuit.library.data_preparation.state_preparation import StatePreparation

def create_hhl_circ(real_powers,B,max_eigval,min_eigval,gen_nodes,tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc,num_time_slices=3):

    """
    Creates a quantum circuit to perform HHL.

    real_powers (list | numpy.array): Contains real powers of all nodes
    B (list of lists | numpy.array with dimension 2): Susceptance matrix of all nodes
    max_eigval (float): Upper bound for maximum eigenvalue of B
    min_eigval (float): Lower bound for minimum eigenvalue of B
    gen_nodes (QuantumRegister): Contains qubits each representing a generator node
    tot_nodes (QuantumRegister): Contains qubits where each basis state represents a node
    stat_prep_anc (QuantumRegister): Contains one qubit used to prepare the initial state
    hhl_phase_reg (QuantumRegister): Register of qubits for storing eigenvalues during HHL
    hhl_anc (QuantumRegister): Contains one qubit used for eigenvalue inversion during HHL
    num_time_slices (int): Number of time slices used for trotterized hamiltonian evolution

    Returns:
    hhl_circ (QuantumCircuit): Quantum circuit that performs HHL

    """

    hhl_circ=QuantumCircuit(gen_nodes,tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc, name="hhl_circ")

    # If the circuit is already stored in a file, then just read the file
 
    with open("circuit_ID.json") as f:
        # This json file contains descriptions of circuits as keys and their
        # respective filenames as values
        circuit_IDs=json.load(f)

    # Circuit qpy files are stored in ../circuits
    current_dir=os.getcwd()
    circuits_dir=os.path.join(current_dir, "circuits")
    
    # Key describing the circuit used in json file to identify circuits
    circuit_key="HHL_"+str(real_powers)+"_"+str(B)+"_"+str(max_eigval)+"_"+str(min_eigval)+\
                "_"+str(len(hhl_phase_reg))+"_"+str(num_time_slices)

    print("Check if HHL circuit already exists")

    circuit_ID=False

    try:
        circuit_ID=circuit_IDs[circuit_key]
        print("Circuit exists. Circuit ID:", circuit_ID)
    except Exception as e:
        print(e)

    if circuit_ID:
        # Read the file to get the circuit and return it
        with open(os.path.join(circuits_dir,circuit_ID+'.qpy'), 'rb') as fd:
            circuit = qpy_serialization.load(fd)[0]
            hhl_circ.compose(circuit, inplace=True)
        return hhl_circ
    
    
    print("Constructing HHL Circuit")

    # Rescaling and resizing

    # Normalize the real_powers vector. This is the state we often call |b> when explaining HHL
    real_powers_norm=np.linalg.norm(np.array(real_powers))
    real_powers=np.array(real_powers)/real_powers_norm

    # Number of extra dimensions we need to make our vector length a power of 2
    extra_dim=2**len(tot_nodes)-len(real_powers)

    # Make real_powers look more like a statevector
    real_powers=np.concatenate((real_powers, np.array([0 for _ in range(extra_dim)])))
    B=np.array(B)
    print("Initial B:", B)

    # Scale B such that the eigenvalues are scaled in such a way that the largest bitstring (111...111)
    # in the phase register represents the max of 1/C and max_eigval
    B=B*2*pi/max_eigval

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

    def qft_dagger(n):
        """n-qubit QFTdagger the first n qubits of hhl_phase_reg"""
        # DO forget the Swaps!
        # for qubit in range(n//2):
        #     qc.swap(qubit, n-qubit-1)
        for j in range(n):
            for m in range(j):
                hhl_circ_temp.cp(-pi/float(2**(j-m)), hhl_phase_reg[m], hhl_phase_reg[j])
            hhl_circ_temp.h(hhl_phase_reg[j])

    qft_dagger(len(hhl_phase_reg))

    hhl_circ.compose(hhl_circ_temp, inplace=True)

    # Conditioned rotations

    hhl_circ.compose(construct_asin_x_inv_circuit(len(hhl_phase_reg),4,2,max_eigval/min_eigval), [q for q in hhl_phase_reg]+[hhl_anc[0]], inplace=True)
    hhl_circ.x(hhl_anc[0])

    # Uncompute QPE to unentangle the ancillas

    hhl_circ.compose(hhl_circ_temp.inverse(), inplace=True)

    circuit_ID=str(len(os.listdir(circuits_dir)))

    circuit_IDs[circuit_key]=circuit_ID
    
    with open("circuit_ID.json", 'w', encoding='utf-8') as f:
        json.dump(circuit_IDs, f, ensure_ascii=False, indent=4)
    

    print("Length of untranspiled HHL", len(hhl_circ))

    print("Transpiling HHL circuit for storage")
    simulator = Aer.get_backend('aer_simulator')
    hhl_circ_transpiled=transpile(hhl_circ, simulator)
    
    with open(os.path.join(circuits_dir,circuit_ID+'.qpy'), 'wb') as fd:
        print("Length of transpiled HHL", len(hhl_circ_transpiled))
        qpy_serialization.dump(hhl_circ_transpiled, fd)

    return(hhl_circ_transpiled)