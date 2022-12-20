import numpy as np
from math import pi
from qiskit import QuantumCircuit, Aer, transpile, QuantumRegister, ClassicalRegister, assemble
from qiskit.circuit import ParameterVector
# from qiskit.circuit.library.data_preparation.state_preparation import prepare_state

from qiskit.opflow import PauliTrotterEvolution, StateFn, PauliExpectation
from qiskit.opflow import CircuitSampler, PauliOp
from qiskit.opflow import I, X, Y, Z, Zero, One, Plus, Minus

from amp_est import real_amp_est
from taylor_precomputation import construct_asin_x_inv_circuit, construct_exp_k_abs_cos_circuit

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
    qft_rotations_0(circuit, n)

def create_hhl_circ(real_powers,B,max_eigval,C,gen_nodes,tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc):

    hhl_circ=QuantumCircuit(gen_nodes,tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc)
    hhl_circ_temp=QuantumCircuit(gen_nodes,tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc)

    # Rescaling and resizing

    real_powers_norm=np.linalg.norm(np.array(real_powers))
    real_powers=np.array(real_powers)/real_powers_norm

    extra_dim=2**len(tot_nodes)-len(real_powers)

    real_powers=np.concatenate(real_powers, np.zeros((extra_dim,1)))
    B=B*2*pi*(2**(len(hhl_phase_reg)-1))/(max_eigval*(2**len(hhl_phase_reg)))

    B=np.concatenate(B, np.zeros((extra_dim, len(list(B)))))
    B=np.concatenate(B, np.zeros((len(list(B)), extra_dim)), axis=1)
    for i in range(B_extra_dim):
        B[-i][-i]=1

    # State prep

    hhl_circ.isometry(real_powers,[],tot_nodes)
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
        hhl_circ.mcx([gen_nodes[i]]+[tot_nodes],state_prep_anc[0])
        # Undo the shuffling we did earlier.
        for j in range(len(tot_nodes)):
            if i%(2**(j+1))<2**j:
                hhl_circ.x(tot_nodes[j])

    hhl_circ.x(gen_nodes)

    # Paulinomial decomposition

    H=Zero
    for i in range(len(tot_nodes)-1):
        H=H^Zero

    paulis=[I,X,Y,Z]

    for i in range(4**tot_nodes):
        term=paulis[i%4]
        i=i//4
        for j in range(tot_nodes-1):
            term=term^paulis[i%4]
            i=i//4
        a=sum([((term.to_matrix())@(-B))[k][k] for k in range(2**tot_nodes)])/(2**tot_nodes)
        H+=(a*term)

    # Create evolution circuit

    evolution_op = (H).exp_i()

    trotterized_op = PauliTrotterEvolution(
                    trotter_mode='trotter',
                    reps=num_time_slices).convert(evolution_op)
    U=trotterized_op.to_circuit().to_gate()
    CU=U.control(2)

    # QPE

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
        # Do forget the Swaps!
        # for qubit in range(n//2):
        #     qc.swap(qubit, n-qubit-1)
        for j in range(n):
            for m in range(j):
                hhl_circ_temp.cp(-pi/float(2**(j-m)), hhl_phase_reg[m], hhl_phase_reg[j])
            qc.h(j)

    qft_dagger(len(hhl_phase_reg))

    hhl_circ.compose(hhl_circ_temp, inplace=True)

    # Conditioned rotations

    hhl_circ.compose(construct_asin_x_inv_circuit(len(hhl_phase_reg),4,C), [q for q in hhl_phase_reg]+[hhl_anc[0]], inplace=True)

    # Uncompute QPE

    hhl_circ.compose(hhl_circ_temp.inverse(), inplace=True)

    return(hhl_circ)

def create_QAOA_ansatz(
    timestep_count, gen_node_count, real_powers, hhl_phase_qubit_count, qadc_qubit_count,
    running_costs, on_off_costs, line_costs, B, max_eigval, C,
    gamma_values, beta_values
    ):

    if len(beta_values)!=len(gamma_values):
        raise
    no_layers=len(gamma_values)
    gamma=ParameterVector(no_layers)
    beta=ParameterVector(no_layers)

    gen_nodes=[QuantumRegister(gen_node_count) for i in range(timestep_count)]
    tot_nodes=QuantumRegister(int(np.ceil(np.log2(len(real_powers)+1))))
    state_prep_anc=QuantumRegister(1)
    hhl_phase_reg=QuantumRegister(hhl_phase_qubit_count)
    hhl_anc=QuantumRegister(1)
    qadc_reg=QuantumRegister(qadc_qubit_count)
    qadc_anc=QuantumRegister(1)

    output_reg=[ClassicalRegister(gen_node_count) for i in range(timestep_count)]

    qc=QuantumCircuit(gen_nodes,tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc,qadc_reg,qadc_anc,output_reg)

    hhl_circ=create_hhl_circ(real_powers,B,max_eigval,C,gen_nodes[0],tot_nodes,state_prep_anc,hhl_phase_reg,hhl_anc)

    qc.h(gen_nodes)

    for layer_index in range(no_layers):

        # Cost layer

        # Running costs
        for t in range(timestep_count):
            for i in range(gen_node_count):
                qc.rz(gamma[layer_index]*running_costs[i], gen_nodes[t][i])

        # On off costs
        for t in range(timestep_count-1):
            for i in range(gen_node_count):
                # On costs
                qc.x(gen_nodes[t][i])
                qc.crz(gamma[layer_index]*on_off_costs[0][i], gen_nodes[t][i], gen_nodes[t+1][i])
                qc.x(gen_nodes[t][i])

                qc.x(gen_nodes[t+1][i])
                qc.crz(gamma[layer_index]*on_off_costs[0][i], gen_nodes[t][i], gen_nodes[t+1][i])
                qc.x(gen_nodes[t+1][i])

        # Transmission Costs
        exp_k_abs_cos_circuit=construct_exp_k_abs_cos_circuit(qadc_qubit_count,4,gamma[layer_index])
        for i in range(gen_node_count):
            for j in range(i):
                C_L=line_costs[i][j]
                if C_L:
                    for t in range(timestep_count):
                        # Move theta_i to 0th component of statevector, also update j accordingly to track the position of theta_j
                        for k in range(tot_nodes):
                            if i%(2**(k+1))>=2**k:
                                hhl_circ.x(tot_nodes[k])
                                if j%(2**(k+1))>=2**k:
                                    j-=2**k
                                else:
                                    j+=2**k
                        # Move theta_j to 2**k-th component
                        k=tot_nodes-1
                        while True:
                            if j<2**k:
                                k-=1
                                continue
                            break
                        for l in range(k):
                            if j%(2**(l+1))>=2**l:
                                hhl_circ.cx(tot_nodes[k],tot_nodes[l])
                        hhl_circ.h(k)
                        hhl_circ.x(k)

                        qadc_circ=real_amp_est(gen_node_count+len(tot_nodes)+2+hhl_phase_qubit_count,0,hhl_circ,qadc_qubit_count)

                        qc.compose(qadc_circ, [q for q in qadc_reg]+[q for q in gen_nodes[t]]+[q for q in tot_nodes]+[state_prep_anc[0]]+[q for q in hhl_phase_reg]+[hhl_anc[0]]+[qadc_anc[0]])
                        qc.compose(exp_k_abs_cos_circuit, qadc_reg)
                        qc.compose(qadc_circ.inverse(), [q for q in qadc_reg]+[q for q in gen_nodes[t]]+[q for q in tot_nodes]+[state_prep_anc[0]]+[q for q in hhl_phase_reg]+[hhl_anc[0]]+[qadc_anc[0]])

        # Penalty Costs
        for t in range(timestep_count):
            C_P=sum(running_costs)+sum(running_costs)+sum([sum(arr) for arr in line_costs])
            

        # Mixer layer
        for t in range(timestep_count):
            for i in range(gen_node_count):
                qc.rx(beta[layer_index], gen_nodes[t][i])

    for t in range(timestep_count):
        qc.measure(gen_nodes[t], output_reg[t])

    return qc
