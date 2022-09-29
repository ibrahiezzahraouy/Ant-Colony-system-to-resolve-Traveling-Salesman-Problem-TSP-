#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


class Node:
    INDEX = -1

    def __new__(cls, *args, **kwargs):
        """Incrémenter automatiquement un indice correspondant au noeud créé"""
        cls.INDEX += 1
        return super().__new__(cls)

    def __init__(self, x, y, name=None):
        self.x = x
        self.y = y
        self.name = name
        self.index = Node.INDEX

    def __eq__(self, other):
        return self.index == other.index
    
    def affiche_node(self):
        return 'node({},{})'.format(self.x,self.y)


def euclidean_distance(node1, node2):
    return ((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2) ** 0.5


class Edge:
    def __init__(self, node1, node2, distance_function=euclidean_distance):
        self.node1 = node1
        self.node2 = node2
        self.distance = distance_function(node1, node2)
        self.pheromone = 1

    def value(self, alpha, beta, d_mean):
        """ Numérateur de la formule de probabilité """
        return self.pheromone ** alpha * (d_mean / self.distance) ** beta


class Graph:
    def __init__(self, nodes, distance_function=euclidean_distance, seed=None):
        self.nodes = {node.index: node for node in nodes}
        self.edges = {}
        self.rng = np.random.default_rng(seed=seed)

        nodes_index = sorted(self.nodes)
        """On remplit le dictionnaire edge par l'ensemble des lignes"""
        for i, index_node_1 in enumerate(nodes_index):
            for index_node_2 in nodes_index[i + 1:]:
                self.edges[(index_node_1, index_node_2)] = Edge(self.nodes[index_node_1], self.nodes[index_node_2], distance_function)
    
    """récupérer une ligne à partir de ses noeuds"""
    def nodes_to_edge(self, node1, node2):
        return self.edges[min(node1.index, node2.index), max(node1.index, node2.index)]

    """choisir le noeud vers lequel se déplcaer à partir d'un noeud de départ"""
    def select_node(self, current_node, nodes, alpha, beta, d_mean):
        if len(nodes) == 1:
            return nodes[0]

        probabilities = np.array([self.nodes_to_edge(current_node, node).value(alpha, beta, d_mean) for node in nodes])
        probabilities = probabilities / np.sum(probabilities)
        chosen_node = self.rng.choice(nodes, p=probabilities)
        return chosen_node
    
    """Vaporisation du phéromone"""
    def global_update_pheromone(self, rho):
        for edge in self.edges.values():
            edge.pheromone = (1 - rho) * edge.pheromone

    """Récupérer la quantité du phéromone contenue dans une ligne"""
    def retrieve_pheromone(self):
        pheromones = dict()
        for k, edge in self.edges.items():
            pheromones[k] = edge.pheromone

        return pheromones



class Ant:
    def __init__(self, graph, d_mean=1.):
        self.position = None
        self.nodes_to_visit = []
        self.graph = graph
        self.distance = 0
        self.edges_visited = []
        self.path = []
        self.d_mean = d_mean

    """Initialiser la position d'une fourmi dans un graphe"""
    def initialize(self, start):
        self.position = start
        self.nodes_to_visit = [node for node in self.graph.nodes.values() if node != self.position]
        self.distance = 0
        self.edges_visited = []
        self.path = [start]

    """Effectuer une itération : le parcours de tous les noeuds """
    def one_iteration(self, alpha, beta):
        while self.nodes_to_visit:
            chosen_node = self.graph.select_node(self.position, self.nodes_to_visit, alpha, beta, self.d_mean)
            self.nodes_to_visit.remove(chosen_node)
            self.path.append(chosen_node)
            chosen_edge = self.graph.nodes_to_edge(self.position, chosen_node)
            self.edges_visited.append(chosen_edge)
            self.distance += chosen_edge.distance
            self.position = chosen_node

    """Mise à jour de phéromone pour les lignes parcourues"""
    def local_update_pheromone(self, d):
        for edge in self.edges_visited:
            edge.pheromone += d / self.distance



class ACO:
    def __init__(self, graph, seed=None):
        self.graph = graph
        self.ants = []
        self.rng = np.random.default_rng(seed=seed)

    """ la fonction de résolution de problème """
    def solve(self, alpha=1, beta=1, rho=0.1, n_ants=20, n_iterations=10, verbose = None):
        
        #distance moyenne des lignes
        d_mean = np.sum(edge.distance for edge in self.graph.edges.values()) / (len(self.graph.edges))
        
        #distance minimale qui peut être parcourue
        min_distance = d_mean * len(self.graph.nodes)
        
        #Ensemble des fourmis
        self.ants = []
        
        #le trajet optimal
        best_path = None
        
        #liste des noeuds
        starts = list(self.graph.nodes.values())
        
        
        for i in range(n_ants):
            self.ants.append(Ant(self.graph, d_mean))
        
        
        for iteration in range(n_iterations):
            
            #Réaliser le suivi de l'évolution de la distance parcourue par pas de 10 itérations
            if iteration % 10 == 0:
                print('Iteration {}/{} :'.format(iteration, n_iterations), min_distance)
            
            #Initialiser la position des fourmis et effectuer une itération
            for ant in self.ants:
                ant.initialize(starts[self.rng.integers(len(starts))])
                ant.one_iteration(alpha, beta)
            
            #Phénomène d'évaporation
            self.graph.global_update_pheromone(rho)
            
            #Mise à jour du niveau de phéromone et choix du meilleur trajet 
            for ant in self.ants:
                ant.local_update_pheromone(min_distance / len(self.ants))
                if ant.distance < min_distance:
                    min_distance = ant.distance
                    best_path = ant.path

        # si on veut visualiser le trajet sous forme de liste
        best_path_display=[]
        for node in best_path:
            best_path_display.append(node.affiche_node())
        best_path_display.append(best_path_display[0])
        

        return best_path, min_distance,best_path_display




