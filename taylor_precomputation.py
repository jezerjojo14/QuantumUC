import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, QuantumRegister, ClassicalRegister, assemble
import matplotlib.pyplot as plt
from scipy.misc import derivative
from math import asin, factorial

import json

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
    # print(j, input_expression)
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

    x_expansion=[[-int(x_scale/2)-1, []]]
    x_expansion+=[[2**i*(x_scale)/2**n, [i]] for i in range(n)]

    asin_expansion=[np.prod([(2*(j+1)-1)/(2*(j+1)) for j in range(i)])/(2*i+1) for i in range(no_terms)]
    x_inv_expansion=[(-1)**i * (int(x_scale/2)+1)**(-i-1) for i in range(no_terms)]

    asin_x_inv_expression=[]
    x_inv_expression=[]

    for i in range(len(x_inv_expansion)):
        # print()
        # print()
        print(i)
        # print()
        # print()
        if x_inv_expansion[i]!=0:
            x_inv_expression=add_expressions(x_inv_expression, scalar_mult(x_inv_expansion[i], power_expansion(x_expansion, i)))
    # asin_x_inv_expression=x_inv_expression
    # x_inv_expression=x_expansion
    for i in range(len(asin_expansion)):
        # print()
        # print()
        print(i)
        # print()
        # print()
        if asin_expansion[i]!=0:
            asin_x_inv_expression=add_expressions(asin_x_inv_expression, scalar_mult(asin_expansion[i], power_expansion(x_inv_expression, 2*i+1)))
    print(asin_x_inv_expression)
    return asin_x_inv_expression

def construct_exp_k_cos_circuit(n, no_terms, k):
    qc=QuantumCircuit(n)

    with open('cos_angles.json') as f:
        data = json.load(f)

    if str(n)+"_"+str(no_terms)+"_"+str(k) in data.keys():
        cos_expression0=data[str(n)+"_"+str(no_terms)+"_"+str(k)]
    else:
        cos_expression0=get_cos_expression(n, no_terms)

        with open('cos_angles.json', 'w', encoding='utf-8') as f:
            data[str(n)+"_"+str(no_terms)+"_"+str(k)]={"angles": cos_expression0}
            json.dump(data, f, ensure_ascii=False, indent=4)

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

    print(n, no_terms, c, lambda_max)

    qc=QuantumCircuit(n+1)

    # asin_x_inv_expression0=get_asin_x_inv_expression(n, no_terms, lambda_max)
    with open('asin_inv_angles.json') as f:
        data = json.load(f)
    
    print(data)

    print(data.keys())
    if str(n)+"_"+str(no_terms)+"_"+str(lambda_max) in data.keys():
        asin_x_inv_expression0=data[str(n)+"_"+str(no_terms)+"_"+str(lambda_max)]
    else:
        asin_x_inv_expression0=get_asin_x_inv_expression(n, no_terms, lambda_max)

        with open('asin_inv_angles.json', 'w', encoding='utf-8') as f:
            data[str(n)+"_"+str(no_terms)+"_"+str(lambda_max)]={"angles": asin_x_inv_expression0}
            json.dump(data, f, ensure_ascii=False, indent=4)

    # asin_x_inv_expression0=[[0.15445959699125714, []], [-0.013467854613723407, [0]], [-0.025840088768203193, [1]], [-0.04752743575465538, [2]], [-0.08020614044877576, [3]], [-0.1144490102393349, [4]], [0.0021333936021695936, [0, 1]], [0.004041668715259125, [0, 2]], [0.0072178337331401995, [0, 3]], [0.011113216977662375, [0, 4]], [0.007860598761224758, [1, 2]], [0.014004451268689943, [1, 3]], [0.021390965838493186, [1, 4]], [0.026309725848645386, [2, 3]], [0.03946908782564946, [2, 4]], [0.06584440633789866, [3, 4]], [-0.0004406087890895274, [0, 1, 2]], [-0.0008556586582235611, [0, 1, 3]], [-0.0016632728215736365, [0, 1, 4]], [-0.0016895807883750976, [0, 2, 3]], [-0.00330207212677888, [0, 2, 4]], [-0.0065407448950106995, [0, 3, 4]], [-0.003359478950828468, [1, 2, 3]], [-0.006582018596563612, [1, 2, 4]], [-0.01305130239678407, [1, 3, 4]], [-0.02600716412104716, [2, 3, 4]], [3.517106095215182e-05, [0, 1, 2, 3]], [3.9449882891433426e-05, [0, 1, 2, 4]], [5.370024117731954e-05, [0, 1, 3, 4]], [8.599526708462783e-05, [0, 2, 3, 4]], [0.0001526154258253206, [1, 2, 3, 4]], [-3.461015761544577e-05, [0, 1, 2, 3, 4]]]

    # 4 qubits, 4 terms
    # asin_x_inv_expression0=[[0.9192716190476192, []], [-0.16244696393367156, [0]], [-0.2880756971275781, [1]], [-0.4690310133071251, [2]], [-0.6670045798573393, [3]], [0.06240519228031269, [0, 1]], [0.0965981590395499, [0, 2]], [0.1342501833565635, [0, 3]], [0.16863402104740277, [1, 2]], [0.23716656269920303, [1, 3]], [0.3839416071195871, [2, 3]], [-0.040018083485118455, [0, 1, 2]], [-0.05274967620291303, [0, 1, 3]], [-0.082380459192386, [0, 2, 3]], [-0.1452490166229745, [1, 2, 3]], [0.030015976077933892, [0, 1, 2, 3]]]
    # asin_x_inv_expression0=[[-63087.6667906746, []], [12687.283502785986, [0]], [23049.25959514332, [1]], [38271.986750478594, [2]], [54300.32710813492, [3]], [62366.300049603175, [4]], [63086.35009920635, [5]], [-4268.10157788479, [0, 1]], [-7220.717149536358, [0, 2]], [-10549.67013687749, [0, 3]], [-12467.678329912014, [0, 4]], [-12687.01859876948, [0, 5]], [-13216.789681414763, [1, 2]], [-19251.137516299885, [1, 3]], [-22672.2842883269, [1, 4]], [-23048.789254347485, [1, 5]], [-32216.48566691081, [2, 3]], [-37706.19415283203, [2, 4]], [-38271.19364420573, [2, 5]], [-53605.963541666664, [3, 4]], [-54299.010416666664, [3, 5]], [-62363.666666666664, [4, 5]], [2266.2649238059917, [0, 1, 2]], [3410.1079309930406, [0, 1, 3]], [4159.764267573753, [0, 1, 4]], [4268.008076687654, [0, 1, 5]], [5823.490589911739, [0, 2, 3]], [7054.338254680236, [0, 2, 4]], [7220.591434458892, [0, 2, 5]], [10340.017128090063, [0, 3, 4]], [10549.530560692152, [0, 3, 5]], [12467.783706108728, [0, 4, 5]], [10714.324971516928, [1, 2, 3]], [12928.981417338053, [1, 2, 4]], [13216.59024556478, [1, 2, 5]], [18890.370297749836, [1, 3, 4]], [19250.919855753582, [1, 3, 5]], [22672.74089558919, [1, 4, 5]], [31673.133463541668, [2, 3, 4]], [32216.216145833332, [2, 3, 5]], [37708.647135416664, [2, 4, 5]], [53630.333333333336, [3, 4, 5]], [-1736.4744694530964, [0, 1, 2, 3]], [-2187.299824297428, [0, 1, 2, 4]], [-2266.1832591295242, [0, 1, 2, 5]], [-3307.8539086580276, [0, 1, 3, 4]], [-3410.0080416202545, [0, 1, 3, 5]], [-4159.15921831131, [0, 1, 4, 5]], [-5665.626546621323, [0, 2, 3, 4]], [-5823.338840007782, [0, 2, 3, 5]], [-7052.219389915466, [0, 2, 4, 5]], [-10324.697973251343, [0, 3, 4, 5]], [-10440.405654907227, [1, 2, 3, 4]], [-10714.055450439453, [1, 2, 3, 5]], [-12923.476623535156, [1, 2, 4, 5]], [-18851.675659179688, [1, 3, 4, 5]], [-31547.5390625, [2, 3, 4, 5]], [1662.712013244629, [0, 1, 2, 3, 4]], [1736.4404907226562, [0, 1, 2, 3, 5]], [2190.599105834961, [0, 1, 2, 4, 5]], [3327.6333770751953, [0, 1, 3, 4, 5]], [5725.481216430664, [0, 2, 3, 4, 5]], [10584.244140625, [1, 2, 3, 4, 5]], [-1606.6291809082031, [0, 1, 2, 3, 4, 5]]]

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
                if len(bits)>=1:
                    qc.mcry(c*angle, qc.qbit_argument_conversion(bits), qc.qbit_argument_conversion(n)[0])
            else:
                i+=1

    return qc

def construct_exp_k_abs_cos_circuit(n, no_terms, k):
    qc=QuantumCircuit(n)

    cos_expression0=get_cos_expression(n, no_terms)

    print(cos_expression0)

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

    asin_x_inv_expression0=get_asin_x_inv_expression(n, no_terms, 8)

    #
    # exp_asin=(construct_asin_x_inv_circuit(n, no_terms, 1, 8)).to_gate()
    #
    # inp_reg=QuantumRegister(n)
    # out_reg=QuantumRegister(1)
    # out_meas=ClassicalRegister(1)
    # x=[]
    y=[]
    # for i in range(2**n):
    #     qc=QuantumCircuit(inp_reg,out_reg,out_meas)
    #     for j in range(n):
    #         if i%(2**(j+1))>=2**j:
    #             qc.x(inp_reg[j])
    #     qc.append(exp_asin, list(inp_reg)+list(out_reg))
    #     qc.measure(out_reg,out_meas)
    #     simulator = Aer.get_backend('qasm_simulator')
    #     result = simulator.run(transpile(qc, simulator)).result()
    #     counts = result.get_counts(qc)
    #     print(counts)
    #     if "1" not in counts.keys():
    #         counts["1"]=0
    #     x+=[8*i/2**n]
    #     y+=[counts['1']/1024]
    #
    # plt.scatter(x, y)
    # plt.show()

    # asin_x_inv_expression0=[[-63087.6667906746, []], [12687.283502785986, [0]], [23049.25959514332, [1]], [38271.986750478594, [2]], [54300.32710813492, [3]], [62366.300049603175, [4]], [63086.35009920635, [5]], [-4268.10157788479, [0, 1]], [-7220.717149536358, [0, 2]], [-10549.67013687749, [0, 3]], [-12467.678329912014, [0, 4]], [-12687.01859876948, [0, 5]], [-13216.789681414763, [1, 2]], [-19251.137516299885, [1, 3]], [-22672.2842883269, [1, 4]], [-23048.789254347485, [1, 5]], [-32216.48566691081, [2, 3]], [-37706.19415283203, [2, 4]], [-38271.19364420573, [2, 5]], [-53605.963541666664, [3, 4]], [-54299.010416666664, [3, 5]], [-62363.666666666664, [4, 5]], [2266.2649238059917, [0, 1, 2]], [3410.1079309930406, [0, 1, 3]], [4159.764267573753, [0, 1, 4]], [4268.008076687654, [0, 1, 5]], [5823.490589911739, [0, 2, 3]], [7054.338254680236, [0, 2, 4]], [7220.591434458892, [0, 2, 5]], [10340.017128090063, [0, 3, 4]], [10549.530560692152, [0, 3, 5]], [12467.783706108728, [0, 4, 5]], [10714.324971516928, [1, 2, 3]], [12928.981417338053, [1, 2, 4]], [13216.59024556478, [1, 2, 5]], [18890.370297749836, [1, 3, 4]], [19250.919855753582, [1, 3, 5]], [22672.74089558919, [1, 4, 5]], [31673.133463541668, [2, 3, 4]], [32216.216145833332, [2, 3, 5]], [37708.647135416664, [2, 4, 5]], [53630.333333333336, [3, 4, 5]], [-1736.4744694530964, [0, 1, 2, 3]], [-2187.299824297428, [0, 1, 2, 4]], [-2266.1832591295242, [0, 1, 2, 5]], [-3307.8539086580276, [0, 1, 3, 4]], [-3410.0080416202545, [0, 1, 3, 5]], [-4159.15921831131, [0, 1, 4, 5]], [-5665.626546621323, [0, 2, 3, 4]], [-5823.338840007782, [0, 2, 3, 5]], [-7052.219389915466, [0, 2, 4, 5]], [-10324.697973251343, [0, 3, 4, 5]], [-10440.405654907227, [1, 2, 3, 4]], [-10714.055450439453, [1, 2, 3, 5]], [-12923.476623535156, [1, 2, 4, 5]], [-18851.675659179688, [1, 3, 4, 5]], [-31547.5390625, [2, 3, 4, 5]], [1662.712013244629, [0, 1, 2, 3, 4]], [1736.4404907226562, [0, 1, 2, 3, 5]], [2190.599105834961, [0, 1, 2, 4, 5]], [3327.6333770751953, [0, 1, 3, 4, 5]], [5725.481216430664, [0, 2, 3, 4, 5]], [10584.244140625, [1, 2, 3, 4, 5]], [-1606.6291809082031, [0, 1, 2, 3, 4, 5]]]

    for i in range(2**6):
        output=0
        for term in asin_x_inv_expression0:
            bits=term[1]
            add_value=True
            for j in bits:
                if i%(2**(j+1))<2**j:
                    add_value=False
            if add_value:
                output+=term[0]
        y+=[output]

    plt.plot(y)
    plt.show()
