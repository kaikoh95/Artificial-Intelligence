"""Solving n-queens puzzle using Local Search techniques"""

import itertools

def conflict_count(n):
    """Takes a total assignment for an n-queen problem and 
    returns the number conflicts for that assignment. We 
    define the number of conflicts to be the number of unordered 
    pairs of queens (objects) that threaten (attack) each other. 
    The assignment will be given in the form of a sequence (tuple 
    more specifically). The assignment is a permutation of numbers 
    from 1 to n. The value of n must be inferred from the given assignment.
    """

    count = 0
    for i in range(len(n)):
        for j in range(i + 1, len(n)):
            dx, dy = j - i,  n[j] - n[i]
            if dx == 0 or dy == 0: 
                continue
            
            check = abs(dx / dy)
            
            if check == 1:
                count += 1

    return count
    

def neighbours(n_tuple):
    """Takes a total assignment for an n-queen problem and returns a 
    sequence (list or iterator) of total assignments that are the neighbours 
    of the current assignment. A neighbour is obtained by swapping the position 
    of two numbers in the given permutation.
    
    Like before, the assignment will be given in the form of a sequence (tuple 
    more specifically). The assignment is a permutation of numbers from 1 to n. 
    The value of n must be inferred from the given assignment.
    """
    
    sol = []
    
    for i in range(len(n_tuple)):
        for j in range(len(n_tuple)):
            if j == i: 
                continue
            
            neighbour = []
            
            for k in range(len(n_tuple)):
                if k == i:
                    neighbour.append(n_tuple[j])
                    
                elif k == j:
                    neighbour.append(n_tuple[i])
                    
                else:
                    neighbour.append(n_tuple[k])

            add_tuple = tuple(neighbour)
            
            if add_tuple not in sol:
                sol.append(add_tuple)
                
    return sol
    

def greedy_descent(n_tuple):
    """Takes an initial total assignment for the n-queens problem 
    and iteratively improves the assignment until either a solution 
    is found or a local minimum is reached. Like before, the assignment 
    will be given in the form of a tuple. The assignment is a permutation 
    of numbers from 1 to n. The value of n must be inferred from the given 
    assignment.
    
    In each iteration, the algorithm must print the current assignment and its 
    corresponding number of conflicts. 
    """
    
    greedy = True
    while greedy:
        count = conflict_count(n_tuple)
        print("Assignment:", n_tuple, "Number of conflicts:", count)

        if count == 0:
            print("A solution is found.")
            greedy = False

        else:
            args = sorted(neighbours(n_tuple))
            n_tuple = min(args, key=lambda neighbour: conflict_count(neighbour))
            
            count1 = conflict_count(n_tuple)
    
            if count1 >= count:
                print("A local minimum is reached.")
                greedy = False
