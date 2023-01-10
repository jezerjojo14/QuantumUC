import numpy as np
from scipy import special, linalg

class Node:

    def __init__(self, real_power, cost_prod, cost_on, cost_off):
        self.lines=[]
        self.real_power = real_power
        self.cost_prod = cost_prod
        self.cost_on = cost_on
        self.cost_off = cost_off

    def add_line(self, line):
        if line not in self.lines:
            self.lines+=[line]


class Line:

    def __init__(self, node1, node2, susceptance, cost_per_watt):
        self.node1 = node1
        self.node2 = node2
        self.susceptance = susceptance
        self.cost_per_watt = cost_per_watt

class Grid:

    b=None
    A=None

    def __init__(self, lines, nodes, node_active):
        self.lines = lines
        self.nodes = node
        self.node_active = node_active
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

    def set_active_nodes(self, node_active):
        self.node_active = node_active
        self.construct_b_vector()
        self.construct_A_matrix()
        self.construct_D_matrix()


    def get_line_from_nodes(self, node1, node2):
        for line in node1.lines:
            if line in node2.lines:
                return line
        return None

    def construct_b_vector(self):
        self.b=np.array([node.real_power * int(self.node_active[self.nodes.index(node)]) for node in self.nodes])

    def construct_A_matrix(self):
        self.A=np.zeros([len(self.nodes),len(self.nodes)])
        for i in range(len(self.nodes)):
            if self.nodes[i]==self.slack_node_neighbor:
                self.A[i,i]+=self.avg_susceptance
            for j in range(len(self.nodes)):
                if i==j:
                    continue
                self.A[i,i]+=self.get_line_from_nodes(self.nodes[i], self.nodes[j]).susceptance
                self.A[i,j]-=self.get_line_from_nodes(self.nodes[i], self.nodes[j]).susceptance
        self.eigenvalue_est_A()

    def eigenvalue_est_A(self):
        eig_max_upp_bound=max([self.A[i,i]*2 - int(self.nodes[i]==self.slack_node_neighbor)*self.avg_susceptance for i in range(len(self.nodes))])
        t=np.array([int(i==0) for i in range(len(self.nodes))])
        eig=0
        while True:
            t = (self.A - eig_max_upp_bound*np.eye(len(self.nodes))) @ t
            if abs(linalg.norm(t)-eig)<0.0001:
                print(t/linalg.norm(t))
                break
            eig=linalg.norm(t)
            t=t/eig
        min_eig=-eig+eig_max_upp_bound
        self.A_eig_bounds = (min_eig, eig_max_upp_bound)



class UCProblem:

    def __init__(self, lines, nodes, timestep_count):
        self.nodes = nodes
        self.lines = lines
        self.timestep_count = timestep_count
        self.grid_timesteps = [Grid(lines, nodes, [True for _ in nodes]) for t in timestep_count]

    def compute_cost(self, bitstring):
        if len(bitstring) != self.timestep_count * len(self.nodes):
            raise

        # TODO

    
