"""
Extra classes built on search.py module.
"""


from search import *
from itertools import dropwhile
from math import *
from heapq import *


class DFSFrontier(Frontier):
    """Implements a frontier container appropriate for depth-first
    search."""

    def __init__(self):
        """The constructor takes no argument. It initialises the
        container to an empty list."""
        self.container = []


    def add(self, path):
        self.container.append(path)

    def __iter__(self):
        
        while self.container:
            yield self.container.pop()


class BFSFrontier(Frontier):
    """Implements a frontier container appropriate for breadth-first
    search."""

    def __init__(self):
        """The constructor takes no argument. It initialises the
        container to an empty list."""
        self.container = []


    def add(self, path):
        self.container.append(path)

    def __iter__(self):
        
        while self.container:
            yield self.container.pop(0) 
            

class LCFSFrontier(Frontier):
    """Performs lowest-cost-first search (LCFS)."""
    
    def __init__(self):
        """The constructor takes no argument. It initialises the
        container to an empty Heap priority queue."""
        
        self.container = [] 
        
    def add(self, path):
        path_cost = 0
        for values in path:
            path_cost += values[3]
        heappush(self.container, (path_cost, path))     

    def __iter__(self):
        while self.container:
            yield heappop(self.container)[1]


class FunkyNumericGraph(Graph):
    """A graph where nodes are numbers. A node (number) n leads to n-1 and
    n+2. Nodes that are divisible by 10 are goal nodes."""
    
    def __init__(self, starting_number):
        self.starting_number = starting_number

    def outgoing_arcs(self, tail_node):
        yield Arc(tail_node, head=tail_node-1, label="1down", cost=1)
        yield Arc(tail_node, head=tail_node+2, label="2up", cost=1)
        
    def starting_nodes(self):
        yield self.starting_number

    def is_goal(self, node):
        return node % 10 == 0
  

class OrderedExplicitGraph(ExplicitGraph):
    """Children of each node are ordered in such a way that
    in depth-first search, child nodes are always expanded in 
    alphabetical order."""
    
    def __init__(self, nodes, edges, starting_list, goal_nodes, estimates=None):
        """Initialises an explicit graph.
        Keyword arguments:
        nodes -- a set of nodes
        edges -- a sequence of tuples in the form (tail, head) or 
                     (tail, head, cost)
        starting_list -- the list of starting nodes (states)
        goal_node -- the set of goal nodes (states)
        """

        # A few assertions to detect possible errors in
        # instantiation. These assertions are not essential to the
        # class functionality.
        assert all(tail in nodes and head in nodes for tail, head, *_ in edges)\
           , "An edge must link two existing nodes!"
        assert all(node in nodes for node in starting_list),\
            "The starting_states must be in nodes."
        assert all(node in nodes for node in goal_nodes),\
            "The goal states must be in nodes."

        self.nodes = nodes      
        self.edges = list(edges)
        self.starting_list = starting_list
        self.goal_nodes = goal_nodes
        self.estimates = estimates

    def starting_nodes(self):
        """Returns (via a generator) a sequence of starting nodes."""
        for starting_node in self.starting_list:
            yield starting_node

    def is_goal(self, node):
        """Returns true if the given node is a goal node."""
        return node in self.goal_nodes

    def outgoing_arcs(self, node):
        """Returns a sequence of Arc objects corresponding to all the
        edges in which the given node is the tail node. The label is
        automatically generated."""
        
        
        self.edges.sort(key=lambda x:x[1])
        self.edges.reverse()
        for edge in self.edges:
            if len(edge) == 2:  # if no cost is specified
                tail, head = edge
                cost = 1        # assume unit cost
            else:
                tail, head, cost = edge
            if tail == node:
                yield Arc(tail, head, str(tail) + '->' + str(head), cost)    
    

class LocationGraph(ExplicitGraph):
    """
    Describes a set of nodes and the connection between them on a 2D plane.
    """
    
    def __init__(self, nodes, locations, edges, starting_list, goal_nodes):
        """Initialises an explicit graph.
        Keyword arguments:
        nodes -- a set of nodes
        locations -- dictionary that specifies the location of each node
        edges -- a set of tuples in the form (tail, head), bidirectional
        starting_list -- the list of starting nodes (states)
        goal_node -- the set of goal nodes (states)
        """
        
        self.nodes = nodes
        self.locations = locations
        self.edges = edges
        self.starting_list = starting_list
        self.goal_nodes = goal_nodes
        
        self.edge_list = []
        for edge in edges:
            head, tail = edge
            self.edge_list.append((tail, head)) #adds the opposite direction 
        self.edge_list += edges
        self.edge_list = list(set(self.edge_list))
        self.edge_list.sort(key=lambda x:x[1])
        
    def outgoing_arcs(self, node):
        """Returns a sequence of Arc objects corresponding to all the
        edges in which the given node is the tail node. The label is
        automatically generated."""

        for edge in self.edge_list:
            if len(edge) == 2:  # if no cost is specified
                tail, head = edge
                
                # Euclidean distance between the tail and head nodes
                cost = sqrt((self.locations[tail][0] - self.locations[head][0]) ** 2 + \
                             (self.locations[tail][1] - self.locations[head][1]) ** 2) 
            else:
                tail, head, cost = edge
            if tail == node:
                yield Arc(tail, head, str(tail) + '->' + str(head), cost)  
                

def main():
    
    """# DFS Frontier
    # Example 1
    graph = ExplicitGraph(nodes=set('SAG'),
                          edge_list=[('S','A'), ('S', 'G'), ('A', 'G')],
                          starting_list=['S'],
                          goal_nodes={'G'})
    solutions = generic_search(graph, DFSFrontier())
    solution = next(solutions, None)
    print_actions(solution)
    # Example 2
    graph = ExplicitGraph(nodes=set('SAG'),
                          edge_list=[('S', 'G'), ('S','A'), ('A', 'G')],
                          starting_list=['S'],
                          goal_nodes={'G'})
    solutions = generic_search(graph, DFSFrontier())
    solution = next(solutions, None)
    print_actions(solution)
    #Example 3
    available_flights = ExplicitGraph(
        nodes=['Christchurch', 'Auckland', 
               'Wellington', 'Gold Coast'],
        edge_list=[('Christchurch', 'Gold Coast'),
                   ('Christchurch','Auckland'),
                   ('Christchurch','Wellington'),
                   ('Wellington', 'Gold Coast'),
                   ('Wellington', 'Auckland'),
                   ('Auckland', 'Gold Coast')],
        starting_list=['Christchurch'],
        goal_nodes={'Gold Coast'})
    
    my_itinerary = next(generic_search(available_flights, DFSFrontier()), None)
    print_actions(my_itinerary)     
    """
    """#BFS Frontier
    #Example 1
    graph = OrderedExplicitGraph(nodes=set('SAG'),
                                 edges={('S','A'), ('S', 'G'), ('A', 'G')},
                                 starting_list=['S'],
                                 goal_nodes={'G'})
                                 
    solutions = generic_search(graph, DFSFrontier())
    solution = next(solutions, None)
    print_actions(solution)    
    #Example 2
    graph = OrderedExplicitGraph(nodes=set('SABG'),
                                 edges={('S', 'A'), ('S','B'),
                                        ('B', 'S'), ('A', 'G')},
                                 starting_list=['S'],
                                 goal_nodes={'G'})
    
    solutions = generic_search(graph, DFSFrontier())
    solution = next(solutions, None)
    print_actions(solution)    
    #Example 3
    flights = OrderedExplicitGraph(nodes={'Christchurch', 'Auckland', 
                                      'Wellington', 'Gold Coast'},
                               edges={('Christchurch', 'Gold Coast'),
                                      ('Christchurch','Auckland'),
                                      ('Christchurch','Wellington'),
                                      ('Wellington', 'Gold Coast'),
                                      ('Wellington', 'Auckland'),
                                      ('Auckland', 'Gold Coast')},
                               starting_list=['Christchurch'],
                               goal_nodes={'Gold Coast'})

    my_itinerary = next(generic_search(flights, DFSFrontier()), None)
    print_actions(my_itinerary)
    """
    """#Funky Numeric
    #Example 1 
    graph = FunkyNumericGraph(4)
    for node in graph.starting_nodes():
        print(node)    
    #Example 2
    graph = FunkyNumericGraph(4)
    for arc in graph.outgoing_arcs(7):
        print(arc)    
    #Example 3
    graph = FunkyNumericGraph(3)
    solutions = generic_search(graph, BFSFrontier())
    print_actions(next(solutions))
    print()
    print_actions(next(solutions))    
    #Example 4
    graph = FunkyNumericGraph(3)
    solutions = generic_search(graph, BFSFrontier())
    print_actions(next(dropwhile(lambda path: path[-1].head <= 10, solutions)))    
    """
    """#Location Graph
    #Example 1
    graph = LocationGraph(nodes=set('ABC'),
                      locations={'A': (0, 0),
                                 'B': (3, 0),
                                 'C': (3, 4)},
                      edges={('A', 'B'), ('B','C'),
                             ('C', 'A')},
                      starting_list=['A'],
                      goal_nodes={'C'})


    for arc in graph.outgoing_arcs('A'):
        print(arc)
    
    for arc in graph.outgoing_arcs('B'):
        print(arc)
    
    for arc in graph.outgoing_arcs('C'):
        print(arc)
    #Example 2
    pythagorean_graph = LocationGraph(
        nodes=set("abc"),
        locations={'a': (5, 6),
                   'b': (10,6),
                   'c': (10,18)},
        edges={tuple(s) for s in {'ab', 'ac', 'bc'}},
        starting_list=['a'],
        goal_nodes={'c'})
    
    for arc in pythagorean_graph.outgoing_arcs('a'):
        print(arc)
    #Example 3
    graph = LocationGraph(nodes=set('ABC'),
                          locations={'A': (0, 0),
                                     'B': (3, 0),
                                     'C': (3, 4)},
                          edges={('A', 'B'), ('B','C'),
                                 ('B', 'A'), ('C', 'A')},
                          starting_list=['A'],
                          goal_nodes={'C'})
    
    
    for arc in graph.outgoing_arcs('A'):
        print(arc)
    
    for arc in graph.outgoing_arcs('B'):
        print(arc)
    
    for arc in graph.outgoing_arcs('C'):
        print(arc)    
    """
    #LCFS Frontier 
    #Example 1
    graph = LocationGraph(nodes=set('ABC'),
                          locations={'A': (0, 0),
                                     'B': (3, 0),
                                     'C': (3, 4)},
                          edges={('A', 'B'), ('B','C'),
                                 ('B', 'A'), ('C', 'A')},
                          starting_list=['A'],
                          goal_nodes={'C'})
    
    solution = next(generic_search(graph, LCFSFrontier()))
    print_actions(solution)  
    #Example 2
    graph = LocationGraph(nodes=set('ABC'),
                          locations={'A': (0, 0),
                                     'B': (3, 0),
                                     'C': (3, 4)},
                          edges={('A', 'B'), ('B','C'),
                                 ('B', 'A')},
                          starting_list=['A'],
                          goal_nodes={'C'})
    
    solution = next(generic_search(graph, LCFSFrontier()))
    print_actions(solution)    
    #Example 3
    pythagorean_graph = LocationGraph(
        nodes=set("abc"),
        locations={'a': (5, 6),
                   'b': (10,6),
                   'c': (10,18)},
        edges={tuple(s) for s in {'ab', 'ac', 'bc'}},
        starting_list=['a'],
        goal_nodes={'c'})
    
    solution = next(generic_search(pythagorean_graph, LCFSFrontier()))
    print_actions(solution)    
    #Example 4
    graph = ExplicitGraph(nodes=set('ABCD'),
                          edge_list=[('A', 'B', 2), ('A', 'D', 7),
                                     ('B', 'C', 3), ('C', 'D', 1)],
                          starting_list=['A'],
                          goal_nodes={'D'})
    
    solution = next(generic_search(graph, LCFSFrontier()))
    print_actions(solution)
    #Example 5
    graph = ExplicitGraph(nodes=set('ABCD'),
                          edge_list=[('A', 'D', 7), ('A', 'B', 2),
                                     ('B', 'C', 3), ('C', 'D', 1)],
                          starting_list=['A'],
                          goal_nodes={'D'})
    
    solution = next(generic_search(graph, LCFSFrontier()))
    print_actions(solution)
    
    
        
if __name__ == "__main__":
    main()
            
