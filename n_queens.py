"""N-queens puzzle is the mathematical generalisation of the 
well-known eight queens puzzle. Using non-chess terms, the objective 
is to place n objects on an n×n board (grid), such that no two objects 
are in the same row, column or diagonal. n is alway a positive integer.

Traditionally, the rows and columns of the board are numbered from 1 to 
n where the position (1,1) can be arbitrarily chosen to be any of the 
squares at the four corners of the board. Using this numbering scheme, 
for example, an assignment in which the positions (5,6) and (5,8) are 
occupied or an assignment in which the positions (3,7)  and (7,7) are 
occupied, are not solutions because in these assignments, two objects 
share the same row or column. Also, for example, having an object at 
(2, 4) and another object at (5, 7) is illegal because the two objects 
share the same diagonal. 

Finding solutions for an n queens puzzle can be computationally expensive 
but by using an appropriate representation and search method, the required 
computational effort can be reduced. In the following, we will solve this 
puzzle (for large values of n) using CSP and local search techniques.
"""

import itertools


def conflict_count(n):
    """Takes a total assignment for an n-queen problem and 
    returns the number conflicts for that assignment. We define 
    the number of conflicts to be the number of unordered pairs 
    of queens (objects) that threaten (attack) each other. The 
    assignment will be given in the form of a sequence (tuple more 
    specifically). The assignment is a permutation of numbers from 
    1 to n. The value of n must be inferred from the given assignment.
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
    """Takes a total assignment for an n-queen problem and 
    returns a sequence (list or iterator) of total assignments 
    that are the neighbours of the current assignment. A neighbour 
    is obtained by swapping the position of two numbers in the given 
    permutation.
    
    Like before, the assignment will be given in the form of a sequence 
    (tuple more specifically). The assignment is a permutation of numbers 
    from 1 to n. The value of n must be inferred from the given assignment.
    
    Because of the choice of representation (the permutation of numbers 
    from 1 to n) the concept of neighbourhood in this question is different 
    from that in the example given in the lecture notes. The representation 
    we use does not allow to have repeated numbers in a sequence, therefore 
    we define a neighbouring assignment to be one that can be obtained by 
    swapping the position of two numbers in the current assignment.
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
    """Takes an initial total assignment for the n-queens problem and 
    iteratively improves the assignment until either a solution is found 
    or a local minimum is reached. Like before, the assignment will be 
    given in the form of a tuple. The assignment is a permutation of 
    numbers from 1 to n. The value of n must be inferred from the given 
    assignment. 
    
    In each iteration, the algorithm must print the current assignment 
    and its corresponding number of conflicts.The function greedy_descent 
    must consider all the neighbours of the current assignment and choose 
    one that has the minimum number of conflicts and has fewer conflicts 
    than the current assignment. If there is not such a neighbour, then a 
    local minimum has been reached and the algorithm must report by the 
    following print function. If an assignment (including the initial 
    assignment) yields no conflict, then it is a solution and the algorithm
    must stop (after printing the assignment and the number of conflicts as 
    above and).
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
                
def random_restart(n):
    """Have a random restart whenever the greedy descent 
    algorithm reaches a local minimum. The program can find a 
    solution for large values of n (e.g. n = 50) in a reasonable 
    time. It is assumed that the function greedy_descent returns 
    a solution if one is found or None otherwise.
    """
    
    random.seed(0) # seeding so that the results can be replicated.
    assignment = list(range(1, n+1))
    while not greedy_descent(tuple(assignment)):
        random.shuffle(assignment)                
