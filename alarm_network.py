"""
Solving belief networks and probabilistic inferrence.
"""

import itertools


def joint_prob(network, assignment):
    """Given a belief network and a complete assignment of all the 
    variables in the network, returns the probability of the assignment. 
    The data structure of the network is as described above. The assignment is a 
    dictionary where keys are the variable names and the values are either True or False.
    """
    
    probability = 1

    for i in assignment.keys():
        value = network[i]
        get_parents = []

        for j in value['Parents']:
            get_parents.append(assignment[j])

        parents = tuple(get_parents)
        result = value['CPT'].get(parents)
        sol = 0
        if assignment[i] is True:
            sol = result
        else:
            sol = 1 - result

        probability *= sol

    return probability


def query(network, query_var, evidence):
    """Given a belief network, the name of a variable in the network, 
    and some evidence, returns the posterior distribution of query_var. 
    The parameter network is a belief network. The parameter query_var is the name 
    of the variable we are interested in and is of type string. The parameter evidence 
    is a dictionary whose elements are assignments to some variables in the network; 
    the keys are the name of the variables and the values are Boolean.
    
    The function returns a pair of real numbers where the first element is the probability 
    of query_var being false given the evidence and the second element is the probability 
    of query_var being true given the evidence.
    """
    
    hidden_vars = network.keys() - evidence.keys() - {query_var}  # finds set of hidden variables
    hidden_assignments = {}
    query_true, query_false = 0, 0

    for values in itertools.product((True, False), repeat=len(hidden_vars)):  # obtain all possible hidden assignments
        hidden_assignments = {var: val for var, val in zip(hidden_vars, values)}
        hidden_assignments.update(evidence)

        # solve for False
        hidden_assignments[query_var] = False
        query_false += joint_prob(network, hidden_assignments)

        # solve for True
        hidden_assignments[query_var] = True
        query_true += joint_prob(network, hidden_assignments)

    query_main = query_false + query_true
    false_query = query_false / query_main
    true_query = query_true / query_main

    return false_query, true_query


# Representation of the alarm network for Belief Networks and Probabilistic Inference.

alarm_net = {
    'Burglary': {
        'Parents': [],
        'CPT': {
            (): 0.001
            }},
            
    'Earthquake': {
        'Parents': [],
        'CPT': {
            (): 0.002,
            }},
    'Alarm': {
        'Parents': ['Burglary','Earthquake'],
        'CPT': {
            (True,True): 0.95,
            (True,False): 0.94,
            (False,True): 0.29,
            (False,False): 0.001,
            }},

    'John': {
        'Parents': ['Alarm'],
        'CPT': {
            (True,): 0.9,
            (False,): 0.05,
            }},

    'Mary': {
        'Parents': ['Alarm'],
        'CPT': {
            (True,): 0.7,
            (False,): 0.01,
            }},
    }
