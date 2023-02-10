import numpy as np
from scipy import special, linalg
from scipy.optimize import minimize
from QAOA_ansatz import create_QAOA_ansatz
from qiskit import execute, transpile, Aer

class Node:
    """
    Creates a new node.

    real_power (list of floats): Power generated (demand if negative) by the node for different time steps
                                    All elements should be equal if node is a generator
    cost_prod (float): Cost of keeping a generator node running (0 if load node)
    cost_on (float): Cost of turning a generator node on (0 if load node)
    cost_off (float): Cost of turning a generator node off (0 if load node)

    lines (list of Line objects): Lines connected to the node
    """

    def __init__(self, real_power, cost_prod=0, cost_on=0, cost_off=0):

        self.lines=[]
        self.real_power = real_power
        self.cost_prod = cost_prod
        self.cost_on = cost_on
        self.cost_off = cost_off

    def add_line(self, line):
        if line not in self.lines:
            self.lines+=[line]


class Line:
    """
    Creates a new line.

    node1 (Node): First node it's connected to
    node2 (Node): Second node it's connected to
    susceptance (float): Susceptance of the line
    cost_of_line (float): Cost of using the line per unit power per unit time
    """

    def __init__(self, node1, node2, susceptance, cost_of_line):
        self.node1 = node1
        self.node2 = node2
        self.susceptance = susceptance
        self.cost_of_line = cost_of_line

class Grid:
    """
    Creates a power grid instance.

    time_step (int): The current time step
    node_active (list of booleans): 
    b (list | numpy.array): Po
    """

    def __init__(self, lines, nodes, node_active):
        self.b=None
        self.A=None
        self.time_step=0
        self.lines = lines
        nodes.sort(reverse=True, key=lambda node:node.real_power[0])
        nodes=[nodes[-1]]+nodes[:-1]
        self.nodes = nodes
        self.node_active = node_active
        if len(node_active)!=len(nodes):
            if len(node_active)!=len([node for node in nodes if node.real_power[0]>0]):
                raise
            node_active=[True]+node_active+[True for _ in range(len(nodes)-len(node_active)-1)]
        self.avg_susceptance=0
        for line in lines:
            line.node1.add_line(line)
            line.node2.add_line(line)
            self.avg_susceptance+=line.susceptance
        self.avg_susceptance/=len(lines)
        self.slack_node_neighbor=nodes[0]
        for node in nodes:
            if len(node.lines)>len(self.slack_node_neighbor.lines):
                self.slack_node_neighbor=node
        self.construct_b_vector()
        self.construct_A_matrix()

    def set_timestep(self, t):
        self.time_step=t

    def set_active_nodes(self, node_active):
        node_active = [bool(el) for el in node_active]
        self.node_active=node_active
        if len(node_active)!=len(self.nodes):
            if len(node_active)!=len([node for node in self.nodes if node.real_power[self.time_step]>0]):
                raise
            self.node_active=[True]+node_active+[True for _ in range(len(self.nodes)-len(node_active)-1)]
        # self
        self.construct_b_vector()
        # self.construct_A_matrix()
        # self.construct_D_matrix()


    def get_line_from_nodes(self, node1, node2):
        for line in node1.lines:
            if line in node2.lines:
                return line
        return None

    def construct_b_vector(self):
        self.b=np.array([node.real_power[self.time_step] * int(self.node_active[self.nodes.index(node)]) for node in self.nodes])

    def construct_A_matrix(self):
        self.A=np.zeros([len(self.nodes),len(self.nodes)])
        for i in range(len(self.nodes)):
            if self.nodes[i]==self.slack_node_neighbor:
                self.A[i,i]+=self.avg_susceptance
            for j in range(len(self.nodes)):
                if i==j:
                    continue
                line=self.get_line_from_nodes(self.nodes[i], self.nodes[j])
                if line:
                    self.A[i,i]+=line.susceptance
                    self.A[i,j]-=line.susceptance
        self.eigenvalue_est_A()

    def eigenvalue_est_A(self):
        eig_max_upp_bound=max([self.A[i,i]*2 - int(self.nodes[i]==self.slack_node_neighbor)*self.avg_susceptance for i in range(len(self.nodes))])
        t=np.array([int(i==0) for i in range(len(self.nodes))])
        eig=0
        while True:
            t = (self.A - eig_max_upp_bound*np.eye(len(self.nodes))) @ t
            if abs(linalg.norm(t)-eig)<0.0001:
                # print(t/linalg.norm(t))
                break
            eig=linalg.norm(t)
            t=t/eig
        min_eig=-eig+eig_max_upp_bound
        self.A_eig_bounds = (min_eig, eig_max_upp_bound)



class UCProblem:

    def __init__(self, lines, nodes, timestep_count):
        nodes.sort(reverse=True, key=lambda node:node.real_power[0])
        self.nodes = nodes
        self.gen_nodes = [node for node in nodes if node.real_power[0]>0]
        self.lines = lines
        self.timestep_count = timestep_count
        self.grid_timesteps = Grid(lines, nodes, [True for _ in nodes])
        self.par_qaoa_circ = None

    def compute_cost(self, bitstring):
        if len(bitstring) != self.timestep_count * len(self.nodes):
            raise

        cost=0
        A=self.grid_timesteps.A
        for j in range(self.timestep_count-1):
            for i in range(len(self.gen_nodes)):
                cost+=bitstring[j*len(self.gen_nodes)+i]*self.gen_nodes[i].cost_prod
                cost+=bitstring[j*len(self.gen_nodes)+i]*(1-bitstring[(j+1)*len(self.gen_nodes)+i])*self.gen_nodes[i].cost_off
                cost+=bitstring[(j+1)*len(self.gen_nodes)+i]*(1-bitstring[j*len(self.gen_nodes)+i])*self.gen_nodes[i].cost_on

            self.grid_timesteps.set_timestep(j)
            self.grid_timesteps.set_active_nodes(bitstring[j*len(self.gen_nodes):(j+1)*len(self.gen_nodes)])
            b=self.grid_timesteps.b
            A_inv=linalg.inv(A)
            x=A_inv @ b
            for p in range(len(x)):
                for q in range(p):
                    line=self.grid_timesteps.get_line_from_nodes(self.nodes[p], self.nodes[q])
                    if line:
                        cost+=line.cost_of_line*abs(line.susceptance*(x[p]-x[q]))
        
        return cost
    
    def create_and_store_QAOA_ansatz(self, no_layers=3):

        B=self.grid_timesteps.A
        # self.grid_timesteps.eigenvalue_est_A()
        C, max_eigval=self.grid_timesteps.A_eig_bounds
        line_costs=np.zeros((len(self.nodes), len(self.nodes)))

        for i in range(len(self.nodes)):
            for j in range(i):
                line=self.grid_timesteps.get_line_from_nodes(self.nodes[i], self.nodes[j])
                if line:
                    line_costs[i][j]=line.cost_of_line
                    line_costs[j][i]=line.cost_of_line


        self.par_qaoa_circ=create_QAOA_ansatz(self.timestep_count, len(self.gen_nodes),
        [node.real_power for node in self.nodes], 4, 4, [node.cost_prod for node in self.gen_nodes],
        [[node.cost_on for node in self.gen_nodes],[node.cost_off for node in self.gen_nodes]], line_costs,
        B, max_eigval, C, no_layers)
        self.backend=Aer.get_backend('aer_simulator')
        self.par_qaoa_circ=transpile(self.par_qaoa_circ, self.backend)

    def get_QAOA_circuit_with_set_parameters(self, params):
        if self.par_qaoa_circ==None:
            self.create_and_store_QAOA_ansatz(int(len(params)/2))
        circ=self.par_qaoa_circ.bind_parameters(params)
        return circ
    
    def estimate_circ_cost(self, params):
        print(params)
        circ=self.get_QAOA_circuit_with_set_parameters(params)
        counts = self.backend.run(circ, nshots=512).result().get_counts()
        print(counts)
        cost=0
        for key in counts.keys():
            bitstring="".join(key.split())
            cost+=self.compute_cost(bitstring)*counts[key]
        return cost
        
    
    def find_optimum_solution(self, initial_guess=np.array([0.33,0.66,1,1,0.66,0.33])):
        res=minimize(self.estimate_circ_cost, initial_guess, method='COBYLA', options={"disp":True})
        print(res)


if __name__=="__main__":
    node1=Node([2,2], 2, 1, 1)
    node2=Node([1,1], 1, 2, 1)
    node3=Node([-1.5,-1])
    node4=Node([-1,0])
    line1=Line(node1,node3,1,1)
    line2=Line(node2,node3,1,1)
    line3=Line(node4,node3,1,1)
    
    problem_instance=UCProblem([line1,line2,line3], [node1,node2,node3], 2)
    problem_instance.find_optimum_solution()