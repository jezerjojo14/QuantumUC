import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from QAOA_ansatz import create_QAOA_ansatz
from qiskit import transpile, Aer
from qiskit.algorithms.optimizers import SPSA
from matplotlib import pyplot as plt

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

    instance_names=[]

    def __init__(self, real_power, cost_prod=0, cost_on=0, cost_off=0, label=None):

        self.lines=[]
        self.real_power = real_power
        self.cost_prod = cost_prod
        self.cost_on = cost_on
        self.cost_off = cost_off
        if label:
            if label not in Node.instance_names:
                self.label=label
                Node.instance_names+=[label]
            else:
                i=1
                while label+"_"+str(i) in Node.instance_names:
                    i+=1
                self.label=label+"_"+str(i)
                Node.instance_names+=[self.label]
        else:
            i=1
            while "node_"+str(i) in Node.instance_names:
                i+=1
            self.label="node_"+str(i)
            Node.instance_names+=[self.label]


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
        # nodes.sort(reverse=True, key=lambda node:node.real_power[0])
        # nodes=[nodes[-1]]+nodes[:-1]
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
        node_active = [bool(int(el)) for el in node_active]
        self.node_active=node_active
        if len(node_active)!=len(self.nodes):
            if len(node_active)!=len([node for node in self.nodes if node.real_power[self.time_step]>0]):
                raise
            self.node_active=[True]+node_active+[True for _ in range(len(self.nodes)-len(node_active)-1)]
        self.construct_b_vector()
        

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
    
    def print_opt(self, number_of_sols=3):
        costs={}
        for i in range(2**(self.timestep_count*len(self.gen_nodes))):
            s=bin(i)[2:].zfill(self.timestep_count*len(self.gen_nodes))
            for j in range(self.timestep_count-1):
                j+=1
                s=s[:self.timestep_count*len(self.gen_nodes)-j*len(self.gen_nodes)]+" "+s[self.timestep_count*len(self.gen_nodes)-j*len(self.gen_nodes):]
            costs[s[::-1]]=self.compute_cost(s)
        
        bitstrings=list(costs.keys())
        bitstrings.sort(key=lambda bitstring:costs[bitstring])

        for sol in bitstrings[:number_of_sols]:
            print(sol, "Cost:", costs[sol])

    def compute_cost(self, bitstring, consider_transmission_costs=True):
        bitstring=bitstring[::-1]
        if len(bitstring.split()) != self.timestep_count:
            print(bitstring)
            print(self.timestep_count, self.gen_nodes)
            raise

        cost=0
        A=self.grid_timesteps.A
        sol=bitstring.split()

        # print(sol)
        # print(sol)
        for t in range(self.timestep_count):
            if len(sol[t])!=len(self.gen_nodes):
                raise
            
            gen_power=0
            for i in range(len(self.gen_nodes)):
                gen_power+=int(sol[t][i])*self.gen_nodes[i].real_power[0]
                cost+=int(sol[t][i])*self.gen_nodes[i].cost_prod
                if t<self.timestep_count-1:
                    cost+=int(sol[t][i])*(1-int(sol[(t+1)][i]))*self.gen_nodes[i].cost_off
                    cost+=int(sol[(t+1)][i])*(1-int(sol[t][i]))*self.gen_nodes[i].cost_on
            
            demand=0
            for i in range(len(self.nodes)-len(self.gen_nodes)):
                i+=len(self.gen_nodes)
                demand-=self.nodes[i].real_power[t]
        
            # print("t=",t)
            # print("Power generated:", gen_power)
            # print("Demand:", demand)
            
            if gen_power<demand:

                penalty_cost=sum([node.cost_prod for node in self.gen_nodes])+ \
                    sum([node.cost_on for node in self.gen_nodes]+[node.cost_off for node in self.gen_nodes])
                if consider_transmission_costs:
                    penalty_cost+=sum([line.cost_of_line*sum([node.real_power[0] for node in self.gen_nodes])/len(self.gen_nodes) for line in self.grid_timesteps.lines])
                # print("Penalty")
                # print("Cost before penalty:",cost)
                cost+=penalty_cost
            
            # print()


            self.grid_timesteps.set_timestep(t)
            self.grid_timesteps.set_active_nodes(sol[t])

            if consider_transmission_costs:

                # b changes when timestep and active nodes change
                b=self.grid_timesteps.b
                A_inv=linalg.inv(A)
                x=A_inv @ b
                for p in range(len(x)):
                    for q in range(p):
                        line=self.grid_timesteps.get_line_from_nodes(self.nodes[p], self.nodes[q])
                        if line:
                            cost+=line.cost_of_line*abs(line.susceptance*(x[p]-x[q]))
            
        # print("Final cost:",cost)
        # print()
        # print()
        return cost
    
    def print_A(self):
        B=self.grid_timesteps.A
        labels=[node.label for node in self.nodes]

        print(labels)
        print(B)
    
    def create_and_store_QAOA_ansatz(self, no_layers=3, consider_transmission_costs=True):

        B=self.grid_timesteps.A
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
        B, max_eigval, C, no_layers, consider_transmission_costs=consider_transmission_costs)

        print("Created circuit")

        self.backend=Aer.get_backend('aer_simulator')
        self.par_qaoa_circ=transpile(self.par_qaoa_circ, self.backend)

    def get_QAOA_circuit_with_set_parameters(self, params, consider_transmission_costs=True):
        if self.par_qaoa_circ==None:
            self.create_and_store_QAOA_ansatz(int(len(params)/2), consider_transmission_costs)
        circ=self.par_qaoa_circ.bind_parameters(params)
        return circ
    
    def estimate_circ_cost(self, params):
        circ=self.get_QAOA_circuit_with_set_parameters(params)
        counts = self.backend.run(circ, shots=1024).result().get_counts()
        # print(counts)

        cost=0

        bitstrings=list(counts.keys())
        bitstrings.sort(key=lambda b: counts[b], reverse=True)

        for i in range(len(bitstrings)):
            # # Sqrt to penalize large variance
            # cost+=self.compute_cost(b, True)*(counts[b]**0.5)
            bstring=bitstrings[i]
            cost+=self.compute_cost(bstring, True)*(counts[bstring])
            # cost+=self.compute_cost(bstring, True)*(counts[bstring])*(i+1)**(-1)
        
        # max_counts=0
        # mode_cost=0

        # for b in counts.keys():
        #     if max_counts<counts[b]:
        #         max_counts=counts[b]
        #         mode_cost=self.compute_cost(b, True)
        
        # # Extra weight on mode of dist
        # cost+=mode_cost*512

        return cost
    
    def estimate_circ_cost_without_transmission_costs(self, params):
        print(params)
        circ=self.get_QAOA_circuit_with_set_parameters(params, False)
        counts = self.backend.run(circ, shots=512).result().get_counts()
        print(counts)
        cost=0
        for key in counts.keys():
            cost+=self.compute_cost(key, False)*counts[key]
        return cost
    
    def print_opt_QAOA(self, params, number_of_sol=3):
        circ=self.get_QAOA_circuit_with_set_parameters(params)
        counts = self.backend.run(circ, shots=2048).result().get_counts()
        bitstrings=list(counts.keys())
        bitstrings.sort(key=lambda bitstring: counts[bitstring])

        top_sols=bitstrings[-3:]

        for sol in top_sols[::-1]:
            print(sol[::-1], "Counts:", counts[sol])
        
        print(counts)
        
        x=[]
        y=[]

        bs=[]

        for b in counts.keys():
            bs+=[b]
            x+=[self.compute_cost(b, True)]
            y+=[counts[b]]
        
        plt.bar([b+"\n"+str(x_el) for b,x_el in zip(np.array(bs)[np.argsort(np.array(x))], np.sort(np.array(x)))],np.array(y)[np.argsort(np.array(x))])
        plt.show()

    
    def find_optimum_solution(self, consider_transmission_costs=True, initial_guess=np.array([0.33,0.66,1,1,0.66,0.33])):       
        opt = SPSA(maxiter=1000)
        if consider_transmission_costs:
            res=opt.minimize(self.estimate_circ_cost, initial_guess)
            # res=minimize(self.estimate_circ_cost, initial_guess, method='COBYLA', options={"disp":True})
        else:
            res=opt.minimize(self.estimate_circ_cost_without_transmission_costs, initial_guess)
            # res=minimize(self.estimate_circ_cost_without_transmission_costs, initial_guess, method='COBYLA', options={"disp":True})
        print(res)
        print("Running circuit with ideal parameters:")
        print("Cost is", self.estimate_circ_cost(res.x))
        print("Top three optimal solutions from QAOA:")
        self.print_opt_QAOA(res.x)
        print("Top three optimal solutions actual:")
        self.print_opt()


if __name__=="__main__":
    node1=Node([2,2], 5, 1, 1, "gen1")
    node2=Node([1,1], 1, 2, 1, "gen2")
    node3=Node([-1.5,-1], 0,0,0, "load1")
    node4=Node([-1,0], 0,0,0, "load2")
    line1=Line(node1,node3,1,1)
    line2=Line(node2,node1,1,1)
    line3=Line(node4,node2,1,1)
    line4=Line(node4,node3,1,1)
    
    problem_instance=UCProblem([line1,line2,line3, line4], [node1,node2,node3,node4], 2)
    problem_instance.print_A()
    problem_instance.find_optimum_solution(consider_transmission_costs=True)
    # problem_instance.find_optimum_solution(consider_transmission_costs=True, initial_guess=np.array([0.25,0.5,0.75,1,1,0.75,0.5,0.25]))