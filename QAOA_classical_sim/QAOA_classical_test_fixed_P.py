from problem_formulation import Node, Line, Grid, UCProblem
import numpy as np
from scipy import linalg
from scipy.optimize import minimize, Bounds, brute
from qiskit.algorithms.optimizers import SPSA
from matplotlib import pyplot as plt

fuel1=1.1
gen1rc=[fuel1*30,fuel1*25,fuel1*100]
gen2rc=[50,10,80]
fuel3=1.2
gen3rc=[fuel3*20,fuel3*20,fuel3*70]

# coeffs=[gen1rc,gen2rc,gen3rc]

no_qubits=6

def U_C(x,problem_instance):

    costs=[]

    # no_qubits=problem_instance.timestep_count*len(problem_instance.gen_nodes)

    for i in range(2**no_qubits):
        s=bin(i)[2:].zfill(no_qubits)
        # compute_cost method expects T bitstrings of size n with spaces in between 
        s=s[:3]+" "+s[3:]
        costs+=[problem_instance.compute_cost_QAOA(s)/(10**8)]

    H_C=np.diag(np.array(costs))
    # print("H_C\n",H_C)
    # U_C=lambda x : linalg.expm(1j*x*H_C)
    return linalg.expm(1j*x*H_C)

X_matrix=np.array([[0,1],[1,0]])
RX=lambda x : linalg.expm(1j*x*X_matrix)

def U_M(x):
    U_M=np.array([[1]])
    for _ in range(no_qubits):
        U_M=np.kron(U_M,RX(x))
    return U_M

def QAOA_ansatz_sim(params,problem_instance):
    no_layers=len(params)//2
    gamma=params[:no_layers]
    beta=params[no_layers:]
    state=np.array([1 for _ in range(2**no_qubits)])*(1/(2**(no_qubits/2)))
    for l in range(no_layers):
        state = U_M(beta[l]) @ U_C(gamma[l],problem_instance) @ state
    return state

def est_QAOA_ansatz_cost(params):
    print("Params:",params)
    P=[600,600,400,400,600,600]

    f=lambda coeffs, p: coeffs[0]+coeffs[1]*p+coeffs[2]*p**2
    
    node1=Node([P[0], P[1]], [f(gen1rc,P[0]),f(gen1rc,P[1])], 100, 100, "gen1")
    node2=Node([P[2], P[3]], [f(gen2rc,P[2]),f(gen2rc,P[3])], 100, 100, "gen2")
    node3=Node([P[4], P[5]], [f(gen3rc,P[4]),f(gen3rc,P[5])], 100, 100, "gen3")

    node4=Node([-600,-200], 0,0,0, "load1")
    node5=Node([-900,-500], 0,0,0, "load2")

    line1=Line(node1,node3,0.5,10)
    line2=Line(node2,node1,0.5,10)
    line3=Line(node4,node2,0.5,10)
    line4=Line(node4,node3,0.5,10)
    line5=Line(node4,node5,0.5,10)
    line6=Line(node5,node3,0.5,10)

    problem_instance=UCProblem([line1,line2,line3,line4,line5,line6], [node1,node2,node3,node4,node5], 2)
 
    state=QAOA_ansatz_sim(params,problem_instance)
    cost=0
    for i in range(len(state)):
        s=bin(i)[2:].zfill(no_qubits)
        bstring=s[:3]+" "+s[3:]
        cost+=problem_instance.compute_cost(bstring, True)*(abs(state[i])**2)
    
    print("Cost:", cost)
    print()
    print()
    return cost

def print_opt_QAOA(params, P):

    f=lambda coeffs, p: coeffs[0]+coeffs[1]*p+coeffs[2]*p**2
    
    node1=Node([P[0], P[1]], [f(gen1rc,P[0]),f(gen1rc,P[1])], 100, 100, "gen1")
    node2=Node([P[2], P[3]], [f(gen2rc,P[2]),f(gen2rc,P[3])], 100, 100, "gen2")
    node3=Node([P[4], P[5]], [f(gen3rc,P[4]),f(gen3rc,P[5])], 100, 100, "gen3")

    node4=Node([-600,-200], 0,0,0, "load1")
    node5=Node([-900,-500], 0,0,0, "load2")

    line1=Line(node1,node3,0.5,10)
    line2=Line(node2,node1,0.5,10)
    line3=Line(node4,node2,0.5,10)
    line4=Line(node4,node3,0.5,10)
    line5=Line(node4,node5,0.5,10)
    line6=Line(node5,node3,0.5,10)

    problem_instance=UCProblem([line1,line2,line3,line4,line5,line6], [node1,node2,node3,node4,node5], 2)
 

    state=QAOA_ansatz_sim(params,problem_instance)
    
    x=[]
    y=[]

    bs=[]

    for i in range(len(state)):
        s=bin(i)[2:].zfill(no_qubits)
        bstring=s[:3]+" "+s[3:]
        bs+=[bstring]
        x+=[problem_instance.compute_cost(bstring, True)]
        y+=[(abs(state[i])**2)]
    
    # plt.bar([b+"\n$"+str(int(x_el)) if x_el!=10**15 else b for b,x_el in zip(np.array(bs)[np.argsort(np.array(x))], np.sort(np.array(x)))],np.array(y)[np.argsort(np.array(x))])
    plt.bar([b+"\n$"+str(int(x_el)) if problem_instance.check_valid(b, True) else b for b,x_el in zip(np.array(bs)[np.argsort(np.array(x))], np.sort(np.array(x)))],np.array(y)[np.argsort(np.array(x))])
    # plt.bar([b+"\n$"+str(int(x_el)) if False else b for b,x_el in zip(np.array(bs)[np.argsort(np.array(x))], np.sort(np.array(x)))],np.array(y)[np.argsort(np.array(x))])
    plt.xticks(rotation='vertical')
    plt.show()

# def find_optimum_solution(initial_guess=np.array([600,600,400,326,500,373, 0.25,0.5,0.75,1,1,0.75,0.5,0.25])):       
# def find_optimum_solution(initial_guess=np.array([599,599,399,326,500,373, 0.25,0.5,0.75,1,1,0.75,0.5,0.25])):       
# def find_optimum_solution(initial_guess=np.array([600,600,400,326,500,373, 0.5,1.0,1.0,0.5])):
def find_optimum_solution(p=3):
    # initial_guess=[-(i+1)/p for i in range(p)]
    # initial_guess+=initial_guess[::-1]
    # initial_guess=[int(100*i)/100 for i in initial_guess]
    res = lambda: None;
    res.x=[-0.43185776, -3.51360934, -3.49831964, -1.5786043 , -0.06133411, -0.74387541]
    # bounds=[(150,600),(150,600),(100,400),(100,400),(50,600),(50,600)]
    # bounds=[(600,600),(600,600),(400,400),(400,400),(600,600),(600,600)]
    # bounds+=[(-2,2) for _ in range(len(initial_guess)-6)]
    bounds=[(-10,10) for _ in range(2*p)]
    opt = SPSA()
    # res=opt.minimize(est_QAOA_ansatz_cost, initial_guess, bounds=bounds)
    # res=minimize(est_QAOA_ansatz_cost, initial_guess, method='COBYLA', bounds=bounds, options={'disp': True})
    # res=minimize(est_QAOA_ansatz_cost, initial_guess, method='COBYLA', bounds=bounds, options={'disp': True})
    # res=brute(est_QAOA_ansatz_cost, bounds, Ns=5)
    # res = lambda: None;
    # res.x= [-9.99999999,  4.99999999,  5.01297715, 10.09904565]
    # print(res)
    P=[600,600,400,400,600,600]
    # print("Optimum powers:", [int(p) for p in P])
    # P=[min(max(p,bound[0]),bound[1]) for p,bound in zip(P,bounds[:6])]
    # print("Optimum powers bounded:", [int(p) for p in P])
    print("Running circuit with ideal parameters:")
    # x=list(P)+list(res.x)[6:]
    print("Cost is", est_QAOA_ansatz_cost(res.x))
    print_opt_QAOA(res.x,P)
    print("Top three optimal solutions actual:")


    f=lambda coeffs, p: coeffs[0]+coeffs[1]*p+coeffs[2]*p**2
    
    node1=Node([P[0], P[1]], [f(gen1rc,P[0]),f(gen1rc,P[1])], 100, 100, "gen1")
    node2=Node([P[2], P[3]], [f(gen2rc,P[2]),f(gen2rc,P[3])], 100, 100, "gen2")
    node3=Node([P[4], P[5]], [f(gen3rc,P[4]),f(gen3rc,P[5])], 100, 100, "gen3")

    node4=Node([-600,-200], 0,0,0, "load1")
    node5=Node([-900,-500], 0,0,0, "load2")

    line1=Line(node1,node3,0.5,10)
    line2=Line(node2,node1,0.5,10)
    line3=Line(node4,node2,0.5,10)
    line4=Line(node4,node3,0.5,10)
    line5=Line(node4,node5,0.5,10)
    line6=Line(node5,node3,0.5,10)

    problem_instance=UCProblem([line1,line2,line3,line4,line5,line6], [node1,node2,node3,node4,node5], 2)
 

    problem_instance.print_opt()

find_optimum_solution()

# x=[ 5.99698922e+02,  6.01318442e+02,  3.99685772e+02,  3.25799932e+02,
#   5.01449703e+02,  3.74333082e+02,  1.44790729e+00, -1.14851077e+00,
#  -1.10611557e+00,  2.96375474e-01, -5.38251199e-01, -1.53817968e+00,
#  -2.45662648e-01,  1.91695662e+00]

# P=(x)[:6]
# print("Optimum powers:", [int(p) for p in P])
# print("Running circuit with ideal parameters:")
# print("Cost is", est_QAOA_ansatz_cost(x))
# print_opt_QAOA(x[6:],P)
# print("Top three optimal solutions actual:")


# f=lambda coeffs, p: coeffs[0]+coeffs[1]*p+coeffs[2]*p**2

# node1=Node([P[0], P[1]], [f(gen1rc,P[0]),f(gen1rc,P[1])], 100000, 100000, "gen1")
# node2=Node([P[2], P[3]], [f(gen2rc,P[2]),f(gen2rc,P[3])], 100000, 100000, "gen2")
# node3=Node([P[4], P[5]], [f(gen3rc,P[4]),f(gen3rc,P[5])], 100000, 100000, "gen3")

# node4=Node([-600,-200], 0,0,0, "load1")
# node5=Node([-900,-500], 0,0,0, "load2")

# line1=Line(node1,node3,0.5,10)
# line2=Line(node2,node1,0.5,10)
# line3=Line(node4,node2,0.5,10)
# line4=Line(node4,node3,0.5,10)
# line5=Line(node4,node5,0.5,10)
# line6=Line(node5,node3,0.5,10)

# problem_instance=UCProblem([line1,line2,line3,line4,line5,line6], [node1,node2,node3,node4,node5], 2)


# problem_instance.print_opt()