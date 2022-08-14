import numpy as np

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

    x_expansion=[[-np.pi*(2*np.pi)/2**n, []]]
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

if __name__ == '__main__':
    n=int(input("Enter number of qubits: "))
    no_terms=int(input("Enter number of terms for cos Taylor expansion: "))
    print(get_cos_expression(n, no_terms))
