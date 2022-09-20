import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, QuantumRegister, ClassicalRegister, assemble
import matplotlib.pyplot as plt
from scipy.misc import derivative
from math import asin

#
# def power_expansion(input_expression, power):
#     for term_0 in input_expression:

def asin_x_inv(x):
    return asin(1/x)

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

def get_asin_x_inv_expression(n, no_terms, x_scale=1):

    # no_terms=int(input("Enter number of terms for cos Taylor expansion: "))

    x_expansion=[[-x_scale, []]]
    x_expansion+=[[2**i*(x_scale)/2**n, [i]] for i in range(n)]

    asin_x_inv_expansion=[derivative(f, x_scale, dx=1e-6, i)/factorial(i) for i in range(no_terms)]
    asin_x_inv_expression=[]
    for i in range(len(asin_x_inv_expansion)):
        # print()
        # print()
        print(i)
        # print()
        # print()
        if cos_expansion[i]!=0:
            asin_x_inv_expression=add_expressions(asin_x_inv_expression, scalar_mult(asin_x_inv_expansion[i], power_expansion(x_expansion, i)))
    return asin_x_inv_expression

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

def construct_asin_x_inv_circuit(n, no_terms, c, lambda_max):
    qc=QuantumCircuit(n+1)

    asin_x_inv_expression0=get_asin_x_inv_expression(n, no_terms, lambda_max)

    asin_x_inv_expression=[el for el in asin_x_inv_expression0]

    # print(cos_expression)
    while len(asin_x_inv_expression):
        # apply_gate(0)
        i=0
        current_terms=[]
        while i < len(asin_x_inv_expression):
            term=asin_x_inv_expression[i]
            # print(current_terms)
            if len([el for c_term in current_terms for el in c_term[1]]+term[1])==len(list(set([el for c_term in current_terms for el in c_term[1]]+term[1]))):
                asin_x_inv_expression.pop(i)
                # print(len(cos_expression))
                current_terms+=[term]
                angle=term[0]
                bits=term[1]
                # print(cos_expression)
                qc.mcrx(c*angle, bits, n)
            else:
                i+=1

    return qc

def construct_exp_k_abs_cos_circuit(n, no_terms, k):
    qc=QuantumCircuit(n)

    # cos_expression0=get_cos_expression(n, no_terms)

    # print(cos_expression0)

    cos_expression0=[[1.0018291040136216, []], [-0.0053906909408226395, [0]], [-0.020195285889376233, [1]], [-0.07757656247375054, [2]], [-0.29466302303153946, [3]], [-1.0018286392476121, [4]], [-2.0018291040136216, [5]], [-0.018736138784648117, [0, 1]], [-0.03669880252043465, [0, 2]], [-0.0673464347348485, [0, 3]], [-0.09262669957043501, [0, 4]], [0.010205964268625845, [0, 5]], [-0.07243349540195086, [1, 2]], [-0.13137958833501148, [1, 3]], [-0.17489540698479594, [1, 4]], [0.039410005486145566, [1, 5]], [-0.24689937382913585, [2, 3]], [-0.3051073198481561, [2, 4]], [0.15369702996249213, [2, 5]], [-0.41244422280639353, [3, 4]], [0.5875562419596146, [3, 5]], [2.0018291040136216, [4, 5]], [0.0023088614855641893, [0, 1, 2]], [0.00729082303506509, [0, 1, 3]], [0.021559118795567676, [0, 1, 4]], [0.03776581012786747, [0, 1, 5]], [0.017034095258993943, [0, 2, 3]], [0.04600287898069347, [0, 2, 4]], [0.07384179735594323, [0, 2, 5]], [0.09946015297839916, [0, 3, 4]], [0.1352446587859886, [0, 3, 5]], [0.18582906215342937, [0, 4, 5]], [0.03641045028875116, [1, 2, 3]], [0.09463737447722967, [1, 2, 4]], [0.14562869601753084, [1, 2, 5]], [0.20210744998219907, [1, 3, 4]], [0.2637014184553754, [1, 3, 5]], [0.3507724479636268, [1, 4, 5]], [0.4128105047117083, [2, 3, 4]], [0.4952022698564211, [2, 3, 5]], [0.6116765322726018, [2, 4, 5]], [0.8267166200943916, [3, 4, 5]], [0.004464162536327908, [0, 1, 2, 3]], [0.004758420216324491, [0, 1, 2, 4]], [-0.004837642032817889, [0, 1, 2, 5]], [0.005338217172684293, [0, 1, 3, 4]], [-0.014860492009333474, [0, 1, 3, 5]], [-0.04341033964910962, [0, 1, 4, 5]], [0.0065046973275048655, [0, 2, 3, 4]], [-0.0344918085260493, [0, 2, 3, 5]], [-0.092444889073755, [0, 2, 4, 5]], [-0.19943510443421625, [0, 3, 4, 5]], [0.008842753970982367, [1, 2, 3, 4]], [-0.07354901117373053, [1, 2, 3, 5]], [-0.1900232735899115, [1, 2, 4, 5]], [-0.40506336141170174, [1, 3, 4, 5]], [-0.8267166200943923, [2, 3, 4, 5]], [-0.008322107504902723, [0, 1, 2, 3, 4]], [-0.008720860942507333, [0, 1, 2, 3, 5]], [-0.009288962263276785, [0, 1, 2, 4, 5]], [-0.01034861936414469, [0, 1, 3, 4, 5]], [-0.012434430450113566, [0, 2, 3, 4, 5]], [-0.01658989727098898, [1, 2, 3, 4, 5]], [0.016589897270989036, [0, 1, 2, 3, 4, 5]]]

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

    exp_cos=(construct_exp_k_abs_cos_circuit(n, no_terms, 5)).to_gate()

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
        for j in range(n):
            if i%(2**(j+1))>=2**j:
                qc.x(inp_reg[j])
        qc.rx(-np.pi/2,  out_reg)
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
