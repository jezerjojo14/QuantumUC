import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, QuantumRegister, ClassicalRegister, assemble
import matplotlib.pyplot as plt

#
# def power_expansion(input_expression, power):
#     for term_0 in input_expression:

def add_expressions(exp0, exp1):
    output_expression=exp0+exp1
    repeat=True
    # print("add_expressions called")
    while repeat:
        repeat=False
        l=len(output_expression)
        for i in range(l):
            # print(i)
            for k in range(l-(i+1)):
                k+=i+1
                if output_expression[k][1]==output_expression[i][1]:
                    repeat=True
                    output_expression[i][0]+=output_expression[k][0]
                    output_expression.pop(k)
                    break
            if output_expression[i][0]==0:
                repeat=True
                output_expression.pop(i)
                break
            if repeat:
                break
    return output_expression

def scalar_mult(c, exp):
    exp1=[[c*term[0], term[1]] for term in exp]
    return exp1

def power_expansion(input_expression, j, term=[1,[]]):
    if j==0:
        return [term]
    output_expression=[]
    for term0 in input_expression:
        bits=list(set(term0[1]+term[1]))
        bits.sort()
        output_expression+=power_expansion(input_expression, j-1, [term[0]*term0[0], bits])
        # output_expression=add_expressions(output_expression, power_expansion(input_expression, j-1, [term[0]*term0[0], bits]))
    repeat=True
    # print("power_expansion called")
    while repeat:
        repeat=False
        l=len(output_expression)
        for i in range(l):
            for k in range(l-(i+1)):
                k+=i+1
                if output_expression[k][1]==output_expression[i][1]:
                    repeat=True
                    output_expression[i][0]+=output_expression[k][0]
                    output_expression.pop(k)
                    break
            if output_expression[i][0]==0:
                repeat=True
                output_expression.pop(i)
                break
            if repeat:
                break

    return output_expression

def get_cos_expression(n, no_terms):

    # no_terms=int(input("Enter number of terms for cos Taylor expansion: "))

    x_expansion=[[-np.pi, []]]
    x_expansion+=[[2**i*(2*np.pi)/2**n, [i]] for i in range(n)]

    cos_expansion=[((-1)**(1+i/2)/np.math.factorial(i) if i%2==0 else 0) for i in range(2*no_terms-1)]
    cos_expression=[]
    x_2=power_expansion(x_expansion,2)
    for i in range(len(cos_expansion)):
        # print()
        # print()
        print(i)
        # print()
        # print()
        if cos_expansion[i]!=0:
            cos_expression=add_expressions(cos_expression, scalar_mult(cos_expansion[i], power_expansion(x_2, i/2)))
    return cos_expression

def construct_exp_k_cos_circuit(n, no_terms, k):
    qc=QuantumCircuit(n)

    cos_expression0=get_cos_expression(n, no_terms)

    cos_expression=[el for el in cos_expression0]

    # print(cos_expression)
    while len(cos_expression):
        # apply_gate(0)
        i=0
        current_terms=[]
        while i < len(cos_expression):
            term=cos_expression[i]
            # print(current_terms)
            if len([el for c_term in current_terms for el in c_term[1]]+term[1])==len(list(set([el for c_term in current_terms for el in c_term[1]]+term[1]))):
                cos_expression.pop(i)
                # print(len(cos_expression))
                current_terms+=[term]
                angle=term[0]
                bits=term[1]
                # print(cos_expression)
                if len(bits)==1:
                    qc.p(k*angle, bits[0])
                elif len(bits)==2:
                    qc.cp(k*angle, bits[0], bits[1])
                elif len(bits)>2:
                    qc.mcp(k*angle, [bit for bit in bits[1:]], bits[0])
            else:
                i+=1

    return qc

def construct_exp_k_abs_cos_circuit(n, no_terms, k):
    qc=QuantumCircuit(n)

    cos_expression0=get_cos_expression(n, no_terms)

    cos_expression=[el for el in cos_expression0]

    # print(cos_expression)
    while len(cos_expression):
        # apply_gate(0)
        i=0
        current_terms=[]
        while i < len(cos_expression):
            term=cos_expression[i]
            # print(current_terms)
            if len([el for c_term in current_terms for el in c_term[1]]+term[1])==len(list(set([el for c_term in current_terms for el in c_term[1]]+term[1]))):
                cos_expression.pop(i)
                # print(len(cos_expression))
                current_terms+=[term]
                angle=term[0]
                bits=term[1]
                if n-1 in bits:
                    bits.remove(n-1)
                    if n-2 not in bits:
                        bits+=[n-2]

                # print(cos_expression)
                if len(bits)==1:
                    qc.p(k*angle, bits[0])
                elif len(bits)==2:
                    qc.cp(k*angle, bits[0], bits[1])
                elif len(bits)>2:
                    qc.mcp(k*angle, [bit for bit in bits[1:]], bits[0])
            else:
                i+=1

    return qc

if __name__ == '__main__':
    n=int(input("Enter number of qubits: "))
    no_terms=int(input("Enter number of terms for cos Taylor expansion: "))
    # exp_cos=(construct_exp_k_abs_cos_circuit(n, no_terms, 1)).to_gate()
    exp_cos=(construct_exp_k_cos_circuit(n, no_terms, 1)).to_gate()
    c_exp_cos=exp_cos.control(1)
    inp_reg=QuantumRegister(n)
    out_reg=QuantumRegister(1)
    out_meas=ClassicalRegister(1)
    x=[]
    y=[]
    for i in range(2**n):
        qc=QuantumCircuit(inp_reg,out_reg,out_meas)
        for j in range(n):
            if i%(2**(j+1))>=2**j:
                qc.x(inp_reg[j])
        qc.h(out_reg)
        qc.p(-np.pi/2, out_reg)
        qc.append(c_exp_cos,list(out_reg)+list(inp_reg))
        qc.rx(np.pi/2,  out_reg)
        qc.measure(out_reg,out_meas)
        simulator = Aer.get_backend('qasm_simulator')
        result = simulator.run(transpile(qc, simulator)).result()
        counts = result.get_counts(qc)
        print(counts)
        if "1" not in counts.keys():
            counts["1"]=0
        x+=[2*np.pi*i/2**n]
        y+=[counts['1']/1024]

    plt.scatter(x, y)
    plt.show()
