import itertools, copy 
from csp import *

def generate_and_test(csp):
    """Takes a CSP object and returns an iterable (e.g. list, tuple, set, generator, ...) 
    of solutions. A solution is a complete assignment that satisfies all the constraints.
    """
    
    names, domains = zip(*csp.var_domains.items())
    
    for values in itertools.product(*domains):
        assignment = {x: v for x, v in zip(names, values)}
        
        if all(satisfies(assignment, constraint) for constraint in csp.constraints):
            yield assignment


def arc_consistent(csp):
    """Takes a CSP object and returns a new CSP object that 
    is arc consistent (and also consequently domain consistent).
    """
    
    csp = copy.deepcopy(csp)
    tda = {(x, c) for c in csp.constraints for x in scope(c)}
    
    while tda:
        x, c = tda.pop()
        ys = list(scope(c) - {x})
        new_domain = set()
        
        for xval in csp.var_domains[x]:
            assignment = {x: xval}
            
            for yvals in itertools.product(*[csp.var_domains[y] for y in ys]):
                assignment.update({y: yval for y, yval in zip(ys, yvals)})
                
                if satisfies(assignment, c):
                    new_domain.add(xval)
                    break
                
        if csp.var_domains[x] != new_domain:
            csp.var_domains[x] = new_domain
            
            for cprime in set(csp.constraints) - {c}:
                if x in scope(c):
                    for z in scope(cprime):
                        if x != z:
                            tda.add((z, cprime))
                            
    return csp