from qiskit import QuantumCircuit, Aer, transpile, QuantumRegister, ClassicalRegister, assemble
from math import pi, cos, sin
from qiskit.visualization import plot_histogram
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

def abs_amp_est(n, k, prep_circ, prec):
    data_reg=QuantumRegister(n)
    address_reg=QuantumRegister(n)
    phase_reg=QuantumRegister(prec)
    swaptest_reg=QuantumRegister(1)

    qc=QuantumCircuit(phase_reg, data_reg,address_reg, swaptest_reg)

    psiprep_circ=QuantumCircuit(data_reg, address_reg, swaptest_reg)
    DataPrepGate=prep_circ.to_gate()
    psiprep_circ.append(DataPrepGate, data_reg)
    k_temp=k
    for i in range(n):
        if k%2:
            psiprep_circ.x(address_reg[i])
        k=k//2
    k=k_temp
    psiprep_circ.h(swaptest_reg[0])
    for i in range(len(address_reg)):
        psiprep_circ.cswap(swaptest_reg[0], address_reg[i], data_reg[i])
    psiprep_circ.h(swaptest_reg[0])

    PsiPrepGate=psiprep_circ.to_gate()
    G_circ=QuantumCircuit(data_reg, address_reg, swaptest_reg)

    G_circ.z(swaptest_reg)

    G_circ.append(PsiPrepGate.inverse(), list(data_reg)+list(address_reg)+list(swaptest_reg))
    G_circ.x(list(data_reg)+list(address_reg)+list(swaptest_reg))
    G_circ.mcp(np.pi, list(data_reg)+list(address_reg), swaptest_reg[0])
    G_circ.x(list(data_reg)+list(address_reg)+list(swaptest_reg))
    G_circ.append(PsiPrepGate, list(data_reg)+list(address_reg)+list(swaptest_reg))

    GGate=G_circ.to_gate()
    CGGate=GGate.control(1)

    qc.h(phase_reg)
    qc.append(PsiPrepGate, list(data_reg)+list(address_reg)+list(swaptest_reg))

    for i in range(len(phase_reg)):
        for j in range(2**i):
            qc.append(CGGate, [phase_reg[i]]+list(data_reg)+list(address_reg)+list(swaptest_reg))

    qft_dagger(qc, len(phase_reg))

    return qc

def real_amp_est(n, k, prep_circ, prec):
    data_reg=QuantumRegister(n)
    phase_reg=QuantumRegister(prec)
    swaptest_reg=QuantumRegister(1)

    qc=QuantumCircuit(phase_reg, data_reg, swaptest_reg)

    psiprep_circ=QuantumCircuit(data_reg, swaptest_reg)
    DataPrepGate=prep_circ.to_gate()
    psiprep_circ.h(swaptest_reg)
    psiprep_circ.append(DataPrepGate.control(1), list(swaptest_reg)+list(data_reg))
    psiprep_circ.x(swaptest_reg)
    k_temp=k
    for i in range(n):
        if k%2:
            psiprep_circ.cx(swaptest_reg[0], data_reg[i])
        k=k//2
    k=k_temp
    psiprep_circ.h(swaptest_reg[0])

    print(psiprep_circ)

    PsiPrepGate=psiprep_circ.to_gate()
    G_circ=QuantumCircuit(data_reg, swaptest_reg)

    G_circ.z(swaptest_reg)

    G_circ.append(PsiPrepGate.inverse(), list(data_reg)+list(swaptest_reg))
    G_circ.x(list(data_reg)+list(swaptest_reg))
    G_circ.mcp(np.pi, list(data_reg), swaptest_reg[0])
    G_circ.x(list(data_reg)+list(swaptest_reg))
    G_circ.append(PsiPrepGate, list(data_reg)+list(swaptest_reg))

    print(G_circ)

    GGate=G_circ.to_gate()
    CGGate=GGate.control(1)

    qc.h(phase_reg)
    qc.append(PsiPrepGate, list(data_reg)+list(swaptest_reg))

    for i in range(len(phase_reg)):
        for j in range(2**i):
            qc.append(CGGate, [phase_reg[i]]+list(data_reg)+list(swaptest_reg))

    qft_dagger(qc, len(phase_reg))

    return qc


if __name__ == '__main__':

    prep_circ=QuantumCircuit(2)
    prep_circ.ry(2*pi*random.random(), 0)
    prep_circ.ry(2*pi*random.random(), 1)

    sv = Statevector.from_instruction(prep_circ)

    print("Actual statevector:")
    print(sv)

    prec=5

    abs=[]
    real=[]

    for k in range(1):

        print("Calculating coefficient "+str(k))

        qc=abs_amp_est(2, k, prep_circ, prec)

        creg=ClassicalRegister(prec)
        qc.add_register(creg)
        qc.measure(list(range(prec)), creg)

        sim = Aer.get_backend('aer_simulator')

        qc=transpile(qc, sim)
        job = sim.run(qc)
        result = job.result()
        counts=result.get_counts()

        amp=0

        for bitstring in counts.keys():
            amp+=((-cos(2*pi*int(bitstring,2)/(2**prec)))**0.5)*counts[bitstring]/1024

        abs+=[amp]

        qc=real_amp_est(2, k, prep_circ, prec)

        creg=ClassicalRegister(prec)
        qc.add_register(creg)
        qc.measure(list(range(prec)), creg)

        print(qc)

        sim = Aer.get_backend('aer_simulator')

        qc=transpile(qc, sim)
        job = sim.run(qc)
        result = job.result()
        counts=result.get_counts()

        amp=0

        for bitstring in counts.keys():
            amp+=(-cos(2*pi*int(bitstring,2)/(2**prec)))*counts[bitstring]/1024

        real+=[amp]
    print()
    print()
    print()
    print("Calculated absolute values of statevector coefficients:")
    print(abs)
    print()
    print()
    print("Calculated real values of statevector coefficients:")
    print(real)
