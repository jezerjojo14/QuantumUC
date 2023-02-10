import numpy as np
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from math import asin
import wexpect

import json

def asin_x_inv(x):
    return asin(1/x)

def power_expansion(n, expression, power):
    print("\tPower expansion")
    child = wexpect.spawn('cmd.exe')
    child.expect('>')
    input_line='a.exe '+str(int(n))+' '+' '.join([str((float(el))) for el in expression])+" "+str(int(power))
    # print(input_line)
    child.sendline(input_line)
    child.expect('>', timeout=99999999999999999999999999999)
    data=((child.before).split())[2+len(expression)+1:-1]
    output_expression=[float(el) for el in data]

    return np.array(output_expression)

def get_cos_expression(n, no_terms):
    
    print("Get cos(x) expression")

    x_expansion=np.array([0.0 for _ in range(2**n)])
    x_expansion[0]=-np.pi

    for i in range(n):
        x_expansion[2**i]=2**i*(2*np.pi)/2**n

    cos_expansion=[((-1)**(1+i/2)/np.math.factorial(i) if i%2==0 else 0) for i in range(2*no_terms-1)]
    cos_expression=np.array([0.0 for _ in range(2**n)])
    x_2=power_expansion(n,x_expansion,2)
    for i in range(len(cos_expansion)):
        if cos_expansion[i]!=0:
            cos_expression+=cos_expansion[i]*(power_expansion(n, x_2, i/2))
    return cos_expression

def get_asin_x_inv_expression(n, no_terms, x_scale=1):

    print("Get asin(x^-1) expression")

    x_expansion=np.array([0 for _ in range(2**n)])
    x_expansion[0]=-int(x_scale/2)-1
    for i in range(n):
        x_expansion[2**i]=2**i*(x_scale)/2**n

    asin_expansion=[np.prod([(2*(j+1)-1)/(2*(j+1)) for j in range(i)])/(2*i+1) for i in range(no_terms)]
    x_inv_expansion=[(-1)**i * (int(x_scale/2)+1)**(-i-1) for i in range(no_terms)]

    asin_x_inv_expression=np.array([0.0 for _ in range(2**n)])
    x_inv_expression=np.array([0.0 for _ in range(2**n)])

    for i in range(len(x_inv_expansion)):
        if x_inv_expansion[i]!=0:
            p_exp=(power_expansion(n, x_expansion, i))
            x_inv_expression+=x_inv_expansion[i]*p_exp
    for i in range(len(asin_expansion)):
        if asin_expansion[i]!=0:
            asin_x_inv_expression+=asin_expansion[i]*(power_expansion(n, x_inv_expression, 2*i+1))
    print("asin_x_inv_expression", asin_x_inv_expression)
    return asin_x_inv_expression

def construct_exp_k_cos_circuit(n, no_terms, k):

    print("Constructing kcos(x) circuit")

    qc=QuantumCircuit(n)

    cos_expression=get_cos_expression(n, no_terms)

    terms_dealt=[0 for _ in cos_expression]
    while 0 in terms_dealt:
        i=0
        current_terms=[]
        while i < len(cos_expression):
            if cos_expression[i]==None:
                i+=1
                continue
            if cos_expression[i]==0:
                terms_dealt[i]=1
                i+=1
                continue
            term=[cos_expression[i], [m for m in range(n) if (format(i, 'b').zfill(n))[m]==1]]
            if len([el for c_term in current_terms for el in c_term[1]]+term[1])==len(list(set([el for c_term in current_terms for el in c_term[1]]+term[1]))):
                cos_expression[i]=None
                terms_dealt[i]=1
                current_terms+=[term]
                angle=term[0]
                bits=term[1]
                if len(bits)==1:
                    qc.p(k*angle, bits[0])
                elif len(bits)==2:
                    qc.cp(k*angle, bits[0], bits[1])
                elif len(bits)>2:
                    qc.mcp(k*angle, [bit for bit in bits[1:]], bits[0])
            i+=1

    return qc

def construct_asin_x_inv_circuit(n, no_terms, c, lambda_max):

    print("Constructing asin(x^-1) circuit")

    qc=QuantumCircuit(n+1)

    with open('asin_inv_angles.json') as f:
        data = json.load(f)
    
    
    if str(n)+"_"+str(no_terms)+"_"+str(lambda_max) in data.keys():
        asin_x_inv_expression=data[str(n)+"_"+str(no_terms)+"_"+str(lambda_max)]["angles"]
    else:
        asin_x_inv_expression=get_asin_x_inv_expression(n, no_terms, lambda_max)

        with open('asin_inv_angles.json', 'w', encoding='utf-8') as f:
            data[str(n)+"_"+str(no_terms)+"_"+str(lambda_max)]={"angles": list(asin_x_inv_expression)}
            json.dump(data, f, ensure_ascii=False, indent=4)


    terms_dealt=[0 for _ in asin_x_inv_expression]
    while 0 in terms_dealt:
        i=0
        current_terms=[]
        while i < len(asin_x_inv_expression):
            if asin_x_inv_expression[i]==None:
                i+=1
                continue
            if asin_x_inv_expression[i]==0:
                terms_dealt[i]=1
                asin_x_inv_expression[i]=None
                i+=1
                continue
            term=[asin_x_inv_expression[i], [m for m in range(n) if (format(i, 'b').zfill(n))[m]==1]]
            if len([el for c_term in current_terms for el in c_term[1]]+term[1])==len(list(set([el for c_term in current_terms for el in c_term[1]]+term[1]))):
                asin_x_inv_expression[i]=None
                terms_dealt[i]=1
                current_terms+=[term]
                angle=term[0]
                bits=term[1]
                if len(bits)>=1:
                    qc.mcry(c*angle, qc.qbit_argument_conversion(bits), qc.qbit_argument_conversion(n)[0])
            else:
                i+=1

    return qc

def construct_exp_k_abs_cos_circuit(n, no_terms, k):
    qc=QuantumCircuit(n)

    print("Constructing exp(k|cos(x)|)")


    with open('cos_angles.json') as f:
        data = json.load(f)

    if str(n)+"_"+str(no_terms) in data.keys():
        cos_expression=data[str(n)+"_"+str(no_terms)]["angles"]
    else:
        cos_expression=get_cos_expression(n, no_terms)

        with open('cos_angles.json', 'w', encoding='utf-8') as f:
            data[str(n)+"_"+str(no_terms)+"_"+str(k)]={"angles": list(cos_expression)}
            json.dump(data, f, ensure_ascii=False, indent=4)

    terms_dealt=[0 for _ in cos_expression]
    while 0 in terms_dealt:
        i=0
        current_terms=[]
        while i < len(cos_expression):
            if cos_expression[i]==None:
                i+=1
                continue
            if cos_expression[i]==0:
                terms_dealt[i]=1
                i+=1
                continue
            term=[cos_expression[i], [m for m in range(n) if (format(i, 'b').zfill(n))[m]==1]]
            if len([el for c_term in current_terms for el in c_term[1]]+term[1])==len(list(set([el for c_term in current_terms for el in c_term[1]]+term[1]))):
                cos_expression[i]=None
                terms_dealt[i]=1
                current_terms+=[term]
                angle=term[0]
                bits=term[1]

                # Absolute value
                if n-1 in bits:
                    bits.remove(n-1)
                    if n-2 not in bits:
                        bits+=[n-2]

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

    asin_x_inv_expression=get_asin_x_inv_expression(n, no_terms, 8)

    y=[]

    for i in range(2**6):
        output=0
        for k in range(len(asin_x_inv_expression)):
            term=[asin_x_inv_expression[k], [m for m in range(n) if (format(k, 'b').zfill(n))[m]==1]]
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
