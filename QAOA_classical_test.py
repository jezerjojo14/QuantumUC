from problem_formulation import Node, Line, Grid, UCProblem
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from qiskit.algorithms.optimizers import SPSA
from matplotlib import pyplot as plt

node1=Node([2,2], 5, 1, 1, "gen1")
node2=Node([1,1], 1, 2, 1, "gen2")
node3=Node([-1.5,-1], 0,0,0, "load1")
node4=Node([-1,-0.75], 0,0,0, "load2")
line1=Line(node1,node3,1,1)
line2=Line(node2,node1,1,1)
line3=Line(node4,node2,1,1)
line4=Line(node4,node3,1,1)

problem_instance=UCProblem([line1,line2,line3, line4], [node1,node2,node3,node4], 2)
problem_instance.print_A()

costs=[]

no_qubits=problem_instance.timestep_count*len(problem_instance.gen_nodes)

for i in range(2**no_qubits):
    s=bin(i)[2:].zfill(no_qubits)
    s=s[:2]+" "+s[2:]
    costs+=[problem_instance.compute_cost(s)]

H_C=np.diag(np.array(costs))
print("H_C\n",H_C)
U_C=lambda x : linalg.expm(1j*x*H_C)

X_matrix=np.array([[0,1],[1,0]])
RX=lambda x : linalg.expm(1j*x*X_matrix)

def U_M(x):
    U_M=np.array([[1]])
    for _ in range(no_qubits):
        U_M=np.kron(U_M,RX(x))
    return U_M

def QAOA_ansatz_sim(params):
    no_layers=len(params)//2
    gamma=params[:no_layers]
    beta=params[no_layers:]
    state=np.array([1 for _ in range(2**no_qubits)])*(1/(2**(no_qubits/2)))
    for l in range(no_layers):
        state = U_M(beta[l]) @ U_C(gamma[l]) @ state
    return state

def est_QAOA_ansatz_cost(params):
    state=QAOA_ansatz_sim(params)
    cost=0
    for i in range(len(state)):
        s=bin(i)[2:].zfill(no_qubits)
        bstring=s[:2]+" "+s[2:]
        cost+=problem_instance.compute_cost(bstring, True)*(abs(state[i])**2)
    return cost

def print_opt_QAOA(params, number_of_sol=3):
    state=QAOA_ansatz_sim(params)
    
    x=[]
    y=[]

    bs=[]

    for i in range(len(state)):
        s=bin(i)[2:].zfill(no_qubits)
        bstring=s[:2]+" "+s[2:]
        bs+=[bstring]
        x+=[problem_instance.compute_cost(bstring, True)]
        y+=[(abs(state[i])**2)]
    
    plt.bar([b+"\n"+str(x_el) for b,x_el in zip(np.array(bs)[np.argsort(np.array(x))], np.sort(np.array(x)))],np.array(y)[np.argsort(np.array(x))])
    plt.show()

# def find_optimum_solution(initial_guess=np.array([0.33,0.66,1,1,0.66,0.33])):       
def find_optimum_solution(initial_guess=np.array([0.25,0.5,0.75,1,1,0.75,0.5,0.25])):       
    opt = SPSA(maxiter=1000)
    res=opt.minimize(est_QAOA_ansatz_cost, initial_guess)
    print(res)
    print("Running circuit with ideal parameters:")
    print("Cost is", est_QAOA_ansatz_cost(res.x))
    print_opt_QAOA(res.x)
    print("Top three optimal solutions actual:")
    problem_instance.print_opt()

find_optimum_solution()