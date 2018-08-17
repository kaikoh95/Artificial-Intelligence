"""
Module for Knowledge Base and Query.
"""


import re
from search import *
from extras import *


def clauses(knowledge_base):
    """Takes the string of a knowledge base; returns an iterator for pairs
    of (head, body) for propositional definite clauses in the
    knowledge base. Atoms are returned as strings. The head is an atom
    and the body is a (possibly empty) list of atoms.

    Author: Kourosh Neshatian

    """
    ATOM   = r"[a-z][a-zA-z\d_]*"
    HEAD   = r"\s*(?P<HEAD>{ATOM})\s*".format(**locals())
    BODY   = r"\s*(?P<BODY>{ATOM}\s*(,\s*{ATOM}\s*)*)\s*".format(**locals())
    CLAUSE = r"{HEAD}(:-{BODY})?\.".format(**locals())
    KB     = r"^({CLAUSE})*\s*$".format(**locals())

    assert re.match(KB, knowledge_base)

    for mo in re.finditer(CLAUSE, knowledge_base):
        yield mo.group('HEAD'), re.findall(ATOM, mo.group('BODY') or "")
        

def forward_deduce(kb):
    """Takes the string of a knowledge base containing propositional 
    definite clauses and returns a (complete) set of atoms (strings)
    that can be derived (to be true) from the knowledge base.
    """
    
    model_list = []
    sol_set = set()
    
    for stuff in clauses(kb):
        model_list.append(stuff)
    
    while len(model_list) > 0:
        index = -1
        for i in range(len(model_list)):
            head, body = model_list[i]
            if not body or set(body).issubset(sol_set):
                index = i
                sol_set.add(head)
                break
        if index >= 0:
            model_list.pop(index)
        else:
            model_list.pop()
    return sol_set


class KBGraph(Graph):
    """Poses a knowledge base and a query as a graph. 
    The query will be a set of atoms (strings).
    """
    
    def __init__(self, kb, query):
        self.query = query
        self.model_list = []
        self.sol_set = set()
        
        for stuff in clauses(kb):
            self.model_list.append(stuff)
       
        for model in self.model_list:
            head, body = model
            if not body:
                self.sol_set.add(head)
    
    def outgoing_arcs(self, tail_node):
        for model in self.model_list:
            head, body = model
            if tail_node[0] == head:
                yield Arc(tail_node, body + tail_node[1:], \
                          label=[tail_node] + ["->"] + body + tail_node[1:], cost=0)
        
    def starting_nodes(self):
        return [list(self.query)]

    def is_goal(self, node):
        return len(node) == 1 and set(node).issubset(self.sol_set) 

    
    
def main():
    #KBGraph
    #Example 1
    kb = """
    a :- b, c.
    b :- d, e.
    b :- g, e.
    c :- e.
    d.
    e.
    f :- a,
         g.
    """
    
    query = {'a'}
    if next(generic_search(KBGraph(kb, query), BFSFrontier()), None):
        print("The query is true.")
    else:
        print("The query is not provable.")    
    #Example 2
    kb = """
    all_tests_passed :- program_is_correct.
    all_tests_passed.
    """
    
    query = {'program_is_correct'}
    if next(generic_search(KBGraph(kb, query), BFSFrontier()), None):
        print("The query is true.")
    else:
        print("The query is not provable.")
    #Example 3
    kb = """
    a :- b, c.
    b :- d, e.
    b :- g, e.
    c :- e.
    d.
    e.
    f :- a,
         g.
    """
    
    query = {'a', 'b', 'd'}
    if next(generic_search(KBGraph(kb, query), DFSFrontier()), None):
        print("The query is true.")
    else:
        print("The query is not provable.")
        


if __name__ == "__main__":
    main()
            