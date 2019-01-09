import random
import numpy as np

class Graph():

    def __init__(self, nx_G, p, q):
        self.G = nx_G
        self.p = p
        self.q = q

    def random_walk(self, walk_length, start_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    pre = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(pre, cur)][0], alias_edges[(pre, cur)][1])]
                    walk.append(next)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk(walk_length=walk_length, start_node=node))
        return walks

    def get_alias_edge(self, src, dst):
        G = self.G
        p = self.p
        q = self.q
        probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                probs.append(G[dst][dst_nbr]['weight'])
            else:
                probs.append(G[dst][dst_nbr]['weight'] / q)
        const = sum(probs)
        probs = [float(prob) / const for prob in probs]
        return alias_setup(probs)

    def process_transition_probs(self):
        G = self.G
        alias_nodes = {}
        for node in G.nodes():
            probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            const = sum(probs)
            probs_1 = [float(prob) / const for prob in probs]
            alias_nodes[node] = alias_setup(probs_1)
        alias_edges = {}
        num = 0
        for edge in G.edges():
            num += 1
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

def alias_setup(probs):
    k = len(probs)
    q = np.zeros(k)
    j = np.zeros(k, dtype=np.int)
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = k * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        j[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return j, q

def alias_draw(j, q):
    k = len(j)
    kk = int(np.floor(np.random.rand() * k))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return j[kk]
