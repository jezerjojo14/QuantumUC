import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
import matplotlib.pyplot as plt
import wexpect

import json

def power_expansion(n, expression, power):
    """
    Runs fortran code to take in polynomial expression of binary variables and returns
    an expression for that polynomial to some power n, and reads the output through a terminal.

    n (int): The number of variables
    expression (array/list of floats): Coefficients of all possible terms such that the m-th element is
                                  the coefficient of (x_1^m_1)(x_2^m_2)...(x_n^m_n) where the binary
                                  expansion of m is m_n...m_2m_1.
    power (int): Power to exponentiate the expression to.

    Returns: An expression in the same format of the input raised to the given power
    """

    print("\tPower expansion")
    print("\tPower: ", power)

    # Call fortran program via command line to do the computation
    child = wexpect.spawn('cmd.exe')
    child.expect('>')
    input_line='a.exe '+str(int(n))+' '+' '.join([str((float(el))) for el in expression])+" "+str(int(power))
    child.sendline(input_line)
    child.expect('>', timeout=99999999999999999999999999999)
    data=((child.before).split())[2+len(expression)+1:-1]
    print("Data read:", data)
    output_expression=[float(el) for el in data]
    print("Output expression", output_expression)
    return np.array(output_expression)

def get_cos_expression(n, no_terms):
    """
    Use Taylor expansion of cos(2pi*x) to get a polynomial expression of x expressed in terms
    of its binary digits encoded such that the maximum value x can take (x_i=1 for all i) is
    (2^n - 1)/2^n.

    n (int): Number of bits of x.
    no_terms (int): Number of terms of the Taylor expansion to be used.

    Returns: Polynomial expression for cos(2pi*x) encoded as a list such that the m-th element
             is the coefficient of (x_1^m_1)(x_2^m_2)...(x_n^m_n) where the binary
             expansion of m is m_n...m_2m_1.
    """
    
    print("Get cos(x) expression")

    # Express the binary expansion of x in the same way we express the output 
    # polynomaial as lists of coefficients

    x_expansion=np.array([0.0 for _ in range(2**n)])

    # x = (2*pi/2^n)\sum 2^i x_i  -  pi

    x_expansion[0]=-np.pi

    for i in range(n):
        x_expansion[2**i]=2**i*(2*np.pi)/2**n

    # cos_expansion: Taylor expansion of cos where the coefficient of x^i is stored in the i-th element 
    cos_expansion=[((-1)**(1+i/2)/np.math.factorial(i) if i%2==0 else 0) for i in range(2*no_terms-1)]

    # cos_expression: Final polynomial expression approximating cos(x) where we store the coefficients
    # of every possible term of binary variables
    cos_expression=np.array([0.0 for _ in range(2**n)])

    # Calculate expression of x^2 in terms of binary variables first since cos is an even function
    x_2=power_expansion(n,x_expansion,2)

    # Calculate polynomial expression for each term in cos_expansion and keep adding to
    # cos expression
    for i in range(len(cos_expansion)):
        if cos_expansion[i]!=0:
            cos_expression+=cos_expansion[i]*(power_expansion(n, x_2, i/2))
    return cos_expression

def get_asin_x_inv_expression(n, no_terms, x_scale=1):
    """
    Use Taylor expansion of asin(1/mx) to get a polynomial expression of x expressed in terms
    of its binary digits encoded such that the maximum value x can take (x_i=1 for all i) is
    (2^n - 1)/2^n.

    n (int): Number of bits of x.
    no_terms (int): Number of terms of the Taylor expansion to be used.
    x_scale (float|int): Constant m in expression asin(1/mx); (1/m <= x <=m)

    Returns: Polynomial expression for asin(1/mx) encoded as a list such that the k-th element
             is the coefficient of (x_1^k_1)(x_2^k_2)...(x_n^k_n) where the binary
             expansion of k is k_n ... k_2 k_1.
    """

    print("Get asin(x^-1) expression")

    # Express the binary expansion of x in the same way we express the output 
    # polynomaial as lists of coefficients
    x_expression=np.array([0.0 for _ in range(2**n)])

    # x = (m/2^n)\sum 2^i x_i  -  (floor(m/2) +1)
    x_expression[0]=-int(x_scale/2)-1
    for i in range(n):
        x_expression[2**i]=float(2**(i-n))*(x_scale)

    # List where i-th element is the coefficient of the x^(2i+1) in the Taylor expansion of asin 
    asin_expansion=[np.prod([(2*(j+1)-1)/(2*(j+1)) for j in range(i)])/(2*i+1) for i in range(no_terms)]
    
    # List where i-th element is the coefficient of the x^i in the Taylor expansion of 1/(x+(floor(m/2) +1))
    # Note that if we plug in x_expression here, this becomes the Taylor expansion of 1/x' where 
    # x'=x+(floor(m/2) +1) = (m/2^n)\sum 2^i x_i
    x_inv_expansion=[(-1)**i * (int(x_scale/2)+1)**(-i-1) for i in range(no_terms)]

    asin_x_inv_expression=np.array([0.0 for _ in range(2**n)])
    x_inv_expression=np.array([0.0 for _ in range(2**n)])

    # Get a polynomial expression for 1/mx where x=(1/2^n)\sum 2^i x_i
    for i in range(len(x_inv_expansion)):
        if x_inv_expansion[i]!=0:
            x_inv_expression+=x_inv_expansion[i]*(power_expansion(n, x_expression, i))
    # Plug this expression into the expansion of asin to get the polynomial expression for asin(1/mx)
    for i in range(len(asin_expansion)):
        if asin_expansion[i]!=0:
            asin_x_inv_expression+=asin_expansion[i]*(power_expansion(n, x_inv_expression, 2*i+1))
    return asin_x_inv_expression

def construct_asin_x_inv_circuit(n, no_terms, c, lambda_max):
    """
    Create a QuantumCircuit that does RY rotations required for HHL.

    n (int): Number of bits of x.
    no_terms (int): Number of terms of the Taylor expansion to be used.
    c (float|int): Constant c in expression c*asin(1/mx)
    lambda_max (float|int): Constant m in expression c*asin(1/mx)

    Returns: QuantumCircuit whose first n qubits make up the input and last 1 qubit is the ancilla
    """

    print("Constructing asin(1/(mx)) circuit")

    input_reg=QuantumRegister(n, name="input_reg") 
    ancilla=QuantumRegister(1, name="anc")
    qc=QuantumCircuit(input_reg, ancilla)


    # Check if angles have already been computed and stored in json file. If so, use them.

    with open('asin_inv_angles.json') as f:
        data = json.load(f)
    
    
    if str(n)+"_"+str(no_terms)+"_"+str(lambda_max) in data.keys():
        asin_x_inv_expression=data[str(n)+"_"+str(no_terms)+"_"+str(lambda_max)]["angles"]
    else:
        asin_x_inv_expression=get_asin_x_inv_expression(n, no_terms, lambda_max)

        with open('asin_inv_angles.json', 'w', encoding='utf-8') as f:
            data[str(n)+"_"+str(no_terms)+"_"+str(lambda_max)]={"angles": list(asin_x_inv_expression)}
            json.dump(data, f, ensure_ascii=False, indent=4)


    # Each term in the expression we get is an angle to rotate the ancilla qubit by, controlled on certain
    # input qubits. The order in which we apply these rotations doesn't matter. We might as well pick an
    # order that minimizes depth and packs all the operations as tightly as possible.

    # This list keeps track of which terms in the expression have already been dealt with on the circuit
    terms_dealt=[0 for _ in asin_x_inv_expression]

    # While there are terms that haven't been worked into the circuit, keep running this loop.
    # Each iteration deals with one layer of depth and picks the terms from the expression that maximizes
    # the number of qubits used while making sure none of the terms overlap.
    # This is a greedy algorithm to minimize the depth.
    while 0 in terms_dealt:
        # i is the index of the term that is being looked at. We initialize it to 0
        i=0
        # Keep track of the terms that make up the current layer 
        current_terms=[]
        while i < len(asin_x_inv_expression):
            # If the i-th term has already been dealt with, increment i by 1 and continue
            if terms_dealt[i]==1:
                i+=1
                continue
            # If the i-th coefficient is 0, no operation is necessary. Increment i by 1 and continue. 
            if asin_x_inv_expression[i]==0:
                terms_dealt[i]=1
                i+=1
                continue

            # If we're still in the loop at this point, we have a non trivial term in the expression.
            # Define the variable 'term' as a list with 0-th element being the angle and 1st element
            # being a list of qubit indices that should control the rotation for this term
            term=[asin_x_inv_expression[i], [n-1-m for m in range(n) if (format(i, 'b').zfill(n))[m]=='1']]

            # This if condition checks to make sure there is no repetition in qubits used if you added
            # the qubits from term to the qubits of all the terms in current_terms
            if len([el for c_term in current_terms for el in c_term[1]]+term[1])==len(list(set([el for c_term in current_terms for el in c_term[1]]+term[1]))):
                # If there is no repetition, then add term to current_terms, set terms_dealt to 1
                # to indicate that the i-th term is dealt with now.
                terms_dealt[i]=1
                current_terms+=[term]

                # Apply the RY rotation by the angle indicated by term[0] and controlled on the qubits
                # specified in term[1]
                angle=term[0]
                bits=term[1]
                if len(bits)>=1:
                    qc.mcry(c*angle, qc.qbit_argument_conversion(bits), ancilla[0])
                else:
                    qc.ry(c*angle, ancilla[0])
            i+=1

    return qc

def construct_exp_k_abs_cos_circuit(n, no_terms, k):
    """
    Create a QuantumCircuit that, given |x>, pulls out a phase exp(ik|cos(x)|)

    n (int): Number of bits of x.
    no_terms (int): Number of terms of the Taylor expansion to be used.
    k (float|int): Constant k in expression exp(ik|cos(x)|)

    Returns: QuantumCircuit
    """
    qc=QuantumCircuit(n)

    print("Constructing exp(ik|cos(x)|)")

    # Check if angles have already been computed and stored in json file. If so, use them.

    with open('cos_angles.json') as f:
        data = json.load(f)

    if str(n)+"_"+str(no_terms) in data.keys():
        cos_expression=data[str(n)+"_"+str(no_terms)]["angles"]
    else:
        cos_expression=get_cos_expression(n, no_terms)

        with open('cos_angles.json', 'w', encoding='utf-8') as f:
            data[str(n)+"_"+str(no_terms)+"_"+str(k)]={"angles": list(cos_expression)}
            json.dump(data, f, ensure_ascii=False, indent=4)

    # Each term in the expression we get is a phase that is to be added, controlled on certain
    # qubits. The order in which we apply these phases doesn't matter. We might as well pick an
    # order that minimizes depth and packs all the operations as tightly as possible.

    # This list keeps track of which terms in the expression have already been dealt with on the circuit
    terms_dealt=[0 for _ in cos_expression]

    # While there are terms that haven't been worked into the circuit, keep running this loop.
    # Each iteration deals with one layer of depth and picks the terms from the expression that maximizes
    # the number of qubits used while making sure none of the terms overlap.
    # This is a greedy algorithm to minimize the depth.
    while 0 in terms_dealt:
        # i is the index of the term that is being looked at. We initialize it to 0
        i=0
        # Keep track of the terms that make up the current layer 
        current_terms=[]
        while i < len(cos_expression):
            # If the i-th term has already been dealt with, increment i by 1 and continue
            if terms_dealt[i]==1:
                i+=1
                continue
            # If the i-th coefficient is 0, no operation is necessary. Increment i by 1 and continue. 
            if cos_expression[i]==0:
                terms_dealt[i]=1
                i+=1
                continue

            # If we're still in the loop at this point, we have a non trivial term in the expression.
            # Define the variable 'term' as a list with 0-th element being the phase angle and 1st element
            # being a list of qubit indices that should control the phase for this term
            term=[cos_expression[i], [n-1-m for m in range(n) if (format(i, 'b').zfill(n))[m]=='1']]

            # This if condition checks to make sure there is no repetition in qubits used if you added
            # the qubits from term to the qubits of all the terms in current_terms
            if len([el for c_term in current_terms for el in c_term[1]]+term[1])==len(list(set([el for c_term in current_terms for el in c_term[1]]+term[1]))):
                # If there is no repetition, then add term to current_terms, set terms_dealt to 1
                # to indicate that the i-th term is dealt with now.
                terms_dealt[i]=1
                current_terms+=[term]

                # We will add a phase indicated by term[0] and controlled on the qubits
                # specified in term[1]
                angle=term[0]
                bits=term[1]

                # Replacing the last qubit with the second last one changes the polynomial expression
                # for cos(x) to that of |cos(x)|
                if n-1 in bits:
                    bits.remove(n-1)
                    if n-2 not in bits:
                        bits+=[n-2]

                # Apply the gate to add the phase
                if len(bits)==1:
                    qc.p(k*angle, bits[0])
                elif len(bits)==2:
                    qc.cp(k*angle, bits[0], bits[1])
                elif len(bits)>2:
                    qc.mcp(k*angle, [bit for bit in bits[1:]], bits[0])
            
            i+=1

    return qc
