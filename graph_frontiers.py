"""
Extra classes built on search.py module.
Contains algorithms for searching in AI.
"""


from search import *
from itertools import *
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

            
class AStarFrontier(Frontier):
    """Performs A* search on graphs. An instance of AStarFrontier together
    with an instance of RoutingGraph will be passed to the generic search
    procedure in order to find the lowest cost solution (if one exists),
    from one of the agents to the goal node. Agents have infinite fuel.
    
    Multiple-path pruning is used by using a dictionary to keep track of points
    visited and the fuel left.
    
    This algorithm halts on all valid maps even when there is no solution.
    """    
    
    def __init__(self, map_graph):
        
        self.map_graph = map_graph
        self.container = [] #initialised as empty
        self.visited = {} #contains visited_points : fuel_left for backtrackking
        self.steps_required = count() #using itertools

    def add(self, path):
        
        curr_row, curr_col, curr_fuel = path[-1].head
        pos = (curr_row, curr_col)
        fuel_left = self.visited.get(pos)
        if fuel_left is None or fuel_left < curr_fuel:
            cost = 0
            
            for arc in path:
                cost += arc.cost
                
            cost += self.map_graph.estimated_cost_to_goal(path[-1].head) #account for heuristic
            heappush(self.container, (cost, next(self.steps_required), path))

    def __iter__(self):

        while self.container:
            cost, steps_required, path = heappop(self.container)
            row, col, fuel = path[-1].head
            pos = (row, col)
            fuel_left = self.visited.get(pos)
            
            if fuel_left is None:
                self.visited[pos] = fuel
                yield path
                
            else:
                
                if fuel_left < fuel:
                    self.visited[pos] = fuel
                    yield path                   


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
     

class RoutingGraph(Graph):
    """Initialised by the map string and represents the map
    in the form of a graph by implementing all the required methods
    in the Graph class, including the method estimated_cost_to_goal. 
    Represent the state of the agent by a tuple of the form (row, column, fuel).
    """
    
    def __init__(self, map_str):
        """Initialises the class and processes the map string given."""
        
        self.map1 =  map_str.splitlines()
        self.starting_list = []
        self.outgoing = []
        
        #Objects to declare
        self.gaps = " "
        self.obstacles = "+-|X"
        self.fuel_up = "F"
        self.agents = "S0123456789"
        self.dest = "G"        

        for row in range(len(self.map1)):
            for col in range(len(self.map1[row])):
                obj = self.map1[row][col]
                
                if obj in self.agents:
                    if obj == 'S':
                        fuel = inf
                    else:
                        fuel = int(obj)
                    state = (row, col, fuel)
                    self.starting_list.append(state)
                elif obj in self.dest:
                    self.goal_node = (row, col)
    
    def starting_nodes(self):
        """Returns a list of starting nodes."""
        
        for nodes in self.starting_list:
            yield nodes

    def is_goal(self, node):
        """Returns true if the given node is a goal node."""
        
        row, col, fuel = node
        return (row, col) == self.goal_node
    
    def estimated_cost_to_goal(self, node):
        """Gets the estimated cost to the goal node from a node.
        Generates formula for most dominant (highest value) function 
        that can be computed in constant time. (Manhattan Distance used).
        """
        
        curr_row, curr_col, fuel = node
        goal_row, goal_col = self.goal_node
        row, col = abs(curr_row - goal_row), abs(curr_col - goal_col)
        D, D2 = 2, 2
        formula = D * max(row, col) + (D2 - D) * min(row, col)
        
        return formula
    
    def outgoing_arcs(self, tail):
        """Gets the outgoing arcs for a node"""
        
        directions = [('N' , -1, 0),
                      ('NE', -1, 1),
                      ('E' ,  0, 1),
                      ('SE',  1, 1),
                      ('S' ,  1, 0),
                      ('SW',  1, -1),
                      ('W' ,  0, -1),
                      ('NW', -1, -1)]  
        
        row, col, fuel = tail
        original_pos = self.map1[row][col]
        
        if fuel > 0:
            
            for label, row_change, col_change in directions: 
                new_row = row + row_change
                new_col = col + col_change
                new_pos = self.map1[new_row][new_col]
                head = (new_row, new_col, fuel-1)
                
                if new_pos not in self.obstacles:
                    cost = 2
                    yield Arc(tail, head, label, cost)
                    
        if original_pos in self.fuel_up and fuel < 9:
            fuel = 9
            head = (row, col, fuel) #restores fuel to max capacity
            label = 'Fuel up'
            cost = 5
            yield Arc(tail, head, label, cost)                           


def print_map(map_graph, frontier, solution):
    """Takes three parameters: an instance of RoutingGraph, 
    a frontier object, and a solution (which is a sequence of Arc objects
    that make up a path from a starting position to the goal position).
    
    Then prints a map such that:
    - the position of the walls, obstacles, agents, and the goal point 
      are all unchanged and they are marked by the same set of characters 
      as in the original map string
    - those free spaces (space characters) that have been expanded during 
      the search are marked with a '.' (a period character)
    - those free spaces (spaces characters) that are part of the solution
      (best path to the goal) are marked with '*' (an asterisk character).
    """
    
    '''
    since python strings are immutable, have to
    convert into nested list in order to replace
    gaps with . or *
    '''
    map_list = []
    row, col = 0, 0
    for row in range(len(map_graph.map1)):
        the_row = []
        for col in range(len(map_graph.map1[row])):
            the_row.append(map_graph.map1[row][col])
        map_list.append(the_row)
    
    ptr = str(map_list[row][col]) #temp placeholder for ptr
            
    for pos in frontier.visited.keys():
        row, col = pos
        ptr = str(map_list[row][col])  
        if ptr in map_graph.gaps:
            map_list[row][col] = '.' #path explored but not taken
    
    if solution:
        for route in solution:
            row, col, fuel = route.head
            ptr = str(map_list[row][col])
            if ptr in map_graph.gaps or ptr == '.':
                map_list[row][col] = '*' #path explored and taken
    
    for row in range(len(map_list)):
        for col in range(len(map_list[row])):
            print(map_list[row][col], end='')
        print() 

        
def main():
    """
    Uncomment each part to test for the respective classes.
    """
    
    ## DFS Frontier ##
    """
    # Example 1
    graph = ExplicitGraph(nodes=set('SAG'),
                          edge_list=[('S','A'), ('S', 'G'), ('A', 'G')],
                          starting_list=['S'],
                          goal_nodes={'G'})
    solutions = generic_search(graph, DFSFrontier())
    solution = next(solutions, None)
    print_actions(solution)
    
    print('-----------------------------------------------')
    
    # Example 2
    graph = ExplicitGraph(nodes=set('SAG'),
                          edge_list=[('S', 'G'), ('S','A'), ('A', 'G')],
                          starting_list=['S'],
                          goal_nodes={'G'})
    solutions = generic_search(graph, DFSFrontier())
    solution = next(solutions, None)
    print_actions(solution)
    
    print('-----------------------------------------------')
    
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
    
    print('-----------------------------------------------')
    """
    
    ## BFS Frontier ##
    """
    #Example 1
    graph = OrderedExplicitGraph(nodes=set('SAG'),
                                 edges={('S','A'), ('S', 'G'), ('A', 'G')},
                                 starting_list=['S'],
                                 goal_nodes={'G'})
                                 
    solutions = generic_search(graph, DFSFrontier())
    solution = next(solutions, None)
    print_actions(solution)    
    
    print('-----------------------------------------------')
    
    #Example 2
    graph = OrderedExplicitGraph(nodes=set('SABG'),
                                 edges={('S', 'A'), ('S','B'),
                                        ('B', 'S'), ('A', 'G')},
                                 starting_list=['S'],
                                 goal_nodes={'G'})
    
    solutions = generic_search(graph, DFSFrontier())
    solution = next(solutions, None)
    print_actions(solution)    
    
    print('-----------------------------------------------')
    
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
    
    print('-----------------------------------------------')
    """
    
    ## Funky Numeric ##
    """
    #Example 1 
    graph = FunkyNumericGraph(4)
    for node in graph.starting_nodes():
        print(node)  
    
    print('-----------------------------------------------')
    
    #Example 2
    graph = FunkyNumericGraph(4)
    for arc in graph.outgoing_arcs(7):
        print(arc) 
        
    print('-----------------------------------------------')
    
    #Example 3
    graph = FunkyNumericGraph(3)
    solutions = generic_search(graph, BFSFrontier())
    print_actions(next(solutions))
    print()
    print_actions(next(solutions))    
    
    print('-----------------------------------------------')
    
    #Example 4
    graph = FunkyNumericGraph(3)
    solutions = generic_search(graph, BFSFrontier())
    print_actions(next(dropwhile(lambda path: path[-1].head <= 10, solutions)))    
    
    print('-----------------------------------------------')
    """
    
    ## Location Graph ##
    """
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
        
    print('-----------------------------------------------')
    
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
        
    print('-----------------------------------------------')
    
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
        
    print('-----------------------------------------------')
    """
    
    ## LCFS Frontier ##
    """
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
    
    print('-----------------------------------------------')
    
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
    
    print('-----------------------------------------------')
    
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
    
    print('-----------------------------------------------')
    
    #Example 4
    graph = ExplicitGraph(nodes=set('ABCD'),
                          edge_list=[('A', 'B', 2), ('A', 'D', 7),
                                     ('B', 'C', 3), ('C', 'D', 1)],
                          starting_list=['A'],
                          goal_nodes={'D'})
    
    solution = next(generic_search(graph, LCFSFrontier()))
    print_actions(solution)
    
    print('-----------------------------------------------')
    
    #Example 5
    graph = ExplicitGraph(nodes=set('ABCD'),
                          edge_list=[('A', 'D', 7), ('A', 'B', 2),
                                     ('B', 'C', 3), ('C', 'D', 1)],
                          starting_list=['A'],
                          goal_nodes={'D'})
    
    solution = next(generic_search(graph, LCFSFrontier()))
    print_actions(solution)
    
    print('-----------------------------------------------')
    """
    
    ## Routing Graph ##
    '''
    #Example 1
    map_str = """\
    +-------+
    |  9  XG|
    |X XXX  |
    | S  0F |
    +-------+
    """

    graph = RoutingGraph(map_str)

    print("Starting nodes:", sorted(graph.starting_nodes()))
    print("Outgoing arcs (available actions) at starting states:")
    for s in sorted(graph.starting_nodes()):
        print(s)
        for arc in graph.outgoing_arcs(s):
            print ("  " + str(arc))

    node = (1,1,5)
    print("\nIs {} goal?".format(node), graph.is_goal(node))
    print("Outgoing arcs (available actions) at {}:".format(node))
    for arc in graph.outgoing_arcs(node):
        print ("  " + str(arc))

    node = (1,7,2)
    print("\nIs {} goal?".format(node), graph.is_goal(node))
    print("Outgoing arcs (available actions) at {}:".format(node))
    for arc in graph.outgoing_arcs(node):
        print ("  " + str(arc))

    node = (3,6,5)
    print("\nIs {} goal?".format(node), graph.is_goal(node))
    print("Outgoing arcs (available actions) at {}:".format(node))
    for arc in graph.outgoing_arcs(node):
        print ("  " + str(arc))

    node = (3,6,9)
    print("\nIs {} goal?".format(node), graph.is_goal(node))
    print("Outgoing arcs (available actions) at {}:".format(node))
    for arc in graph.outgoing_arcs(node):
        print ("  " + str(arc))
        
    print('-----------------------------------------------')
    
    #Example 2
    map_str = """\
    +--+
    |GS|
    +--+
    """

    graph = RoutingGraph(map_str)

    print("Starting nodes:", sorted(graph.starting_nodes()))
    print("Outgoing arcs (available actions) at the start:")
    for start in graph.starting_nodes():
        for arc in graph.outgoing_arcs(start):
            print ("  " + str(arc))



    node = (1,1,1)
    print("\nIs {} goal?".format(node), graph.is_goal(node))
    print("Outgoing arcs (available actions) at {}:".format(node))
    for arc in graph.outgoing_arcs(node):
        print ("  " + str(arc))
        
    print('-----------------------------------------------')
    
    #Example 3
    map_str = """\
    +----+
    | X  |
    |XSX |
    | X G|
    +----+
    """

    graph = RoutingGraph(map_str)

    print("Starting nodes:", sorted(graph.starting_nodes()))
    print("Available actions at the start:")
    for s in graph.starting_nodes():
        for arc in graph.outgoing_arcs(s):
            print ("  " + arc.label)
    
    print('-----------------------------------------------')
    '''
    
    ## A* Routing Frontier ##
    '''
    #Example 1
    map_str = """\
    +-------+
    |  F  XG|
    |X XXXX |
    | 2     |
    +-------+
    """

    map_graph = RoutingGraph(map_str)
    frontier = AStarFrontier(map_graph)
    solution = next(generic_search(map_graph, frontier), None)
    print_actions(solution)
    
    print('-----------------------------------------------')
    
    #Example 2
    map_str = """\
    +---------+
    |         |
    |    G    |
    |         |
    +---------+
    """

    map_graph = RoutingGraph(map_str)
    frontier = AStarFrontier(map_graph)
    solution = next(generic_search(map_graph, frontier), None)
    print_actions(solution)
    
    print('-----------------------------------------------')
    
    #Example 3
    map_str = """\
    +----------------+
    |2             F |
    |XX   G 123      |
    |2XXXXXXXXXXXXXX |
    |  F             |
    |           F    |
    +----------------+
    """

    map_graph = RoutingGraph(map_str)
    frontier = AStarFrontier(map_graph)
    solution = next(generic_search(map_graph, frontier), None)
    print_actions(solution)
    
    print('-----------------------------------------------')
    '''
    
    ## Map Visualization ##
    '''
    #Example 1
    map_str = """\
    +------------+
    |            |
    |            |
    |            |
    |    S       |
    |            |
    |            |
    | G          |
    |            |
    +------------+
    """

    map_graph = RoutingGraph(map_str)
    frontier = AStarFrontier(map_graph)
    solution = next(generic_search(map_graph, frontier), None)
    print_map(map_graph, frontier, solution)
    
    print('-----------------------------------------------')
    
    #Example 2
    map_str = """\
    +---------------+
    |    G          |
    |XXXXXXXXXXXX   |
    |           X   |
    |  XXXXXX   X   |
    |  X S  X   X   |
    |  X        X   |
    |  XXXXXXXXXX   |
    |               |
    +---------------+
    """

    map_graph = RoutingGraph(map_str)
    frontier = AStarFrontier(map_graph)
    solution = next(generic_search(map_graph, frontier), None)
    print_map(map_graph, frontier, solution)
    
    print('-----------------------------------------------')
    
    #Example 3
    map_str = """\
    +-------------+
    |         G   |
    | S           |
    |         S   |
    +-------------+
    """

    map_graph = RoutingGraph(map_str)
    frontier = AStarFrontier(map_graph)
    solution = next(generic_search(map_graph, frontier), None)
    print_map(map_graph, frontier, solution)
    
    print('-----------------------------------------------')
    
    #Example 4
    map_str = """\
    +------------+
    |         X  |
    | S       X G|
    |         X  |
    |         X  |
    |         X  |
    +------------+
    """

    map_graph = RoutingGraph(map_str)
    frontier = AStarFrontier(map_graph)
    solution = next(generic_search(map_graph, frontier), None)
    print_map(map_graph, frontier, solution)
    
    print('-----------------------------------------------')
    '''
            
    

if __name__ == "__main__":
    main()
