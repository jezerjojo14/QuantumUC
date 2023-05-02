from qiskit import QuantumCircuit, Aer, transpile, QuantumRegister, ClassicalRegister
from math import pi, cos
import numpy as np
from qiskit.quantum_info import Statevector
import random

def qft_dagger(qc, n):
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-pi/float(2**(j-m)), m, j)
        qc.h(j)

def real_amp_est(n, k, prep_circ, prec):
    """
    Gets the real part of the k-th entry Re(v_k) of the statevector |v> = prep_circ|0>
    and stores arccos(Re(v_k)) in basis encoding.

    n (int): Number of qubits in prep_circ
    k (int): Index of amplitude that we want to estimate
    prep_circ (QuantumCircuit): Circuit that prepares the state we want to
                                conduct amplitude estimation on
    prec (int): Number of qubits we want to use to store arccos(Re(v_k))
    """
    data_reg=QuantumRegister(n)
    phase_reg=QuantumRegister(prec)
    hadamardtest_reg=QuantumRegister(1)

    qc=QuantumCircuit(phase_reg, data_reg, hadamardtest_reg)

    # psiprep_circ is the circuit that prapares the |Psi> state from |0>
    psiprep_circ=QuantumCircuit(data_reg, hadamardtest_reg)

    # DataPrepGate is the gate of prep_circ which prepares the state we want to
    # conduct amplitude estimation on

    DataPrepGate=prep_circ.to_gate()
    psiprep_circ.h(hadamardtest_reg)
    psiprep_circ.append(DataPrepGate.control(1), list(hadamardtest_reg)+list(data_reg))
    psiprep_circ.x(hadamardtest_reg)
    k_temp=k
    for i in range(n):
        if k%2:
            psiprep_circ.cx(hadamardtest_reg[0], data_reg[i])
        k=k//2
    k=k_temp
    psiprep_circ.h(hadamardtest_reg[0])

    print(psiprep_circ)

    PsiPrepGate=psiprep_circ.to_gate()

    # The operator G = R_k Z first applies a Z gate on the anc qubit. Then it performs an
    # operation R_k which takes the |Psi> state to -|Psi> and any orthogonal state to itself

    G_circ=QuantumCircuit(data_reg, hadamardtest_reg)

    G_circ.z(hadamardtest_reg)

    G_circ.append(PsiPrepGate.inverse(), list(data_reg)+list(hadamardtest_reg))
    G_circ.x(list(data_reg)+list(hadamardtest_reg))
    G_circ.mcp(np.pi, list(data_reg), hadamardtest_reg[0])
    G_circ.x(list(data_reg)+list(hadamardtest_reg))
    G_circ.append(PsiPrepGate, list(data_reg)+list(hadamardtest_reg))

    print(G_circ)

    GGate=G_circ.to_gate()
    CGGate=GGate.control(1)

    # Apply phase estimation on G with the initial state |Psi>

    qc.h(phase_reg)
    qc.append(PsiPrepGate, list(data_reg)+list(hadamardtest_reg))

    for i in range(len(phase_reg)):
        for j in range(2**i):
            qc.append(CGGate, [phase_reg[i]]+list(data_reg)+list(hadamardtest_reg))

    qft_dagger(qc, len(phase_reg))

    return qc