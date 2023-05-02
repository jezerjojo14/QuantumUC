from problem_formulation import Node, Line, Grid, UCProblem
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from qiskit.algorithms.optimizers import SPSA
from matplotlib import pyplot as plt

gen1rc=[30,25,100]
gen2rc=[50,10,80]
gen3rc=[20,20,70]

# coeffs=[gen1rc,gen2rc,gen3rc]

no_qubits=6

# def compute_cost(self, bitstring, coeffs, f):
#     bitstring=bitstring[::-1]
#     if len(bitstring.split()) != self.timestep_count:
#         print(bitstring)
#         print(self.timestep_count, self.gen_nodes)
#         raise

#     cost=0
#     A=self.grid_timesteps.A
#     sol=bitstring.split()

#     # print(sol)
#     # print(sol)
#     for t in range(self.timestep_count):
#         if len(sol[t])!=len(self.gen_nodes):
#             raise
        
#         gen_power=0
#         for i in range(len(self.gen_nodes)):
#             gen_power+=int(sol[t][i])*self.gen_nodes[i].real_power[t]
#             cost+=int(sol[t][i])*f(coeffs[i],self.gen_nodes[i].real_power[t])
#             if t<self.timestep_count-1:
#                 cost+=int(sol[t][i])*(1-int(sol[(t+1)][i]))*self.gen_nodes[i].cost_off
#                 cost+=int(sol[(t+1)][i])*(1-int(sol[t][i]))*self.gen_nodes[i].cost_on
        
#         demand=0
#         for i in range(len(self.nodes)-len(self.gen_nodes)):
#             i+=len(self.gen_nodes)
#             demand-=self.nodes[i].real_power[t]
    
#         # print("t=",t)
#         # print("Power generated:", gen_power)
#         # print("Demand:", demand)
        
#         if gen_power<demand:

#             penalty_cost=sum([node.cost_prod for node in self.gen_nodes])+ \
#                 sum([node.cost_on for node in self.gen_nodes]+[node.cost_off for node in self.gen_nodes])
#             penalty_cost+=sum([line.cost_of_line*sum([node.real_power[t] for node in self.gen_nodes])/len(self.gen_nodes) for line in self.grid_timesteps.lines])
#             # print("Penalty")
#             # print("Cost before penalty:",cost)
#             cost+=penalty_cost
        
#         # print()


#         self.grid_timesteps.set_timestep(t)
#         self.grid_timesteps.set_active_nodes(sol[t])

#         # b changes when timestep and active nodes change
#         b=self.grid_timesteps.b
#         A_inv=linalg.inv(A)
#         x=A_inv @ b
#         for p in range(len(x)):
#             for q in range(p):
#                 line=self.grid_timesteps.get_line_from_nodes(self.nodes[p], self.nodes[q])
#                 if line:
#                     cost+=line.cost_of_line*abs(line.susceptance*(x[p]-x[q]))
        
#     # print("Final cost:",cost)
#     # print()
#     # print()
#     return cost

def U_C(x,problem_instance):

    costs=[]

    # no_qubits=problem_instance.timestep_count*len(problem_instance.gen_nodes)

    for i in range(2**no_qubits):
        s=bin(i)[3:].zfill(no_qubits)
        # compute_cost method expects T bitstrings of size n with spaces in between 
        s=s[:3]+" "+s[3:]
        costs+=[problem_instance.compute_cost(s)]

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

def est_QAOA_ansatz_cost(params_tot):
    print("Params:",params_tot)
    P=params_tot[:6]
    params=params_tot[6:]

    f=lambda coeffs, p: coeffs[0]+coeffs[1]*p+coeffs[2]*p**2
    
    node1=Node([P[0], P[1]], [f(gen1rc,P[0]),f(gen1rc,P[1])], 100000, 100000, "gen1")
    node2=Node([P[2], P[3]], [f(gen2rc,P[2]),f(gen2rc,P[3])], 100000, 100000, "gen2")
    node3=Node([P[4], P[5]], [f(gen3rc,P[4]),f(gen3rc,P[5])], 100000, 100000, "gen3")

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
        s=bin(i)[3:].zfill(no_qubits)
        bstring=s[:3]+" "+s[3:]
        cost+=problem_instance.compute_cost(bstring, True)*(abs(state[i])**2)
    
    print("Cost:", cost)
    print()
    print()
    return cost

def print_opt_QAOA(params, P):

    f=lambda coeffs, p: coeffs[0]+coeffs[1]*p+coeffs[2]*p**2
    
    node1=Node([P[0], P[1]], [f(gen1rc,P[0]),f(gen1rc,P[1])], 100000, 100000, "gen1")
    node2=Node([P[2], P[3]], [f(gen2rc,P[2]),f(gen2rc,P[3])], 100000, 100000, "gen2")
    node3=Node([P[4], P[5]], [f(gen3rc,P[4]),f(gen3rc,P[5])], 100000, 100000, "gen3")

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
        s=bin(i)[3:].zfill(no_qubits)
        bstring=s[:3]+" "+s[3:]
        bs+=[bstring]
        x+=[problem_instance.compute_cost(bstring, True)]
        y+=[(abs(state[i])**2)]
    
    plt.bar([b+"\n"+str(int(x_el)) for b,x_el in zip(np.array(bs)[np.argsort(np.array(x))], np.sort(np.array(x)))],np.array(y)[np.argsort(np.array(x))])
    plt.xticks(rotation='vertical')
    plt.show()

def find_optimum_solution(initial_guess=np.array([600,600,400,326,500,373, 0.33,0.66,1,1,0.66,0.33])):       
# def find_optimum_solution(initial_guess=np.array([600,600,400,326,500,373, 0.25,0.5,0.75,1,1,0.75,0.5,0.25])):       
# def find_optimum_solution(initial_guess=np.array([600,600,400,326,500,373, 0.5,1,1,0.5])):
    
    bounds=[(150,600),(150,600),(100,400),(100,400),(50,600),(50,600)]
    bounds+=[(-np.inf,np.inf) for _ in range(len(initial_guess)-6)]
    opt = SPSA()
    res=opt.minimize(est_QAOA_ansatz_cost, initial_guess, bounds=bounds)
    print(res)
    P=(res.x)[:6]
    print("Optimum powers:", [int(p) for p in P])
    print("Running circuit with ideal parameters:")
    print("Cost is", est_QAOA_ansatz_cost(res.x))
    print_opt_QAOA(res.x[6:],P)
    print("Top three optimal solutions actual:")


    f=lambda coeffs, p: coeffs[0]+coeffs[1]*p+coeffs[2]*p**2
    
    node1=Node([P[0], P[1]], [f(gen1rc,P[0]),f(gen1rc,P[1])], 100000, 100000, "gen1")
    node2=Node([P[2], P[3]], [f(gen2rc,P[2]),f(gen2rc,P[3])], 100000, 100000, "gen2")
    node3=Node([P[4], P[5]], [f(gen3rc,P[4]),f(gen3rc,P[5])], 100000, 100000, "gen3")

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