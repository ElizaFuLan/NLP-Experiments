"""
COMS W4705 - Natural Language Processing - Fall 2023
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        n = len(tokens)
        table = [[set() for _ in range(n+1)] for _ in range(n+1)]

        # Fill in the table for terminals
        for i in range(n):
            for lhs, rhs, _ in self.grammar.rhs_to_rules.get((tokens[i],), []):
                table[i][i+1].add(lhs)

        # Fill in the table for longer spans
        for span in range(2, n+1):
            for i in range(n-span+1):
                j = i + span
                for k in range(i+1, j):
                    for rule_list in self.grammar.lhs_to_rules.values():
                        for rule in rule_list:
                            A = rule[0]
                            rhs = rule[1]
                            if len(rhs) == 2:
                                B, C = rhs
                                if B in table[i][k] and C in table[k][j]:
                                    table[i][j].add(A)

        # Check if the start symbol can generate the whole sentence
        return self.grammar.startsymbol in table[0][n]
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        n = len(tokens)
        table = defaultdict(dict)
        probs = defaultdict(dict)

        # Initialize tables with terminal rules
        for i in range(n):
            for lhs, rhs, prob in self.grammar.rhs_to_rules.get((tokens[i],), []):
                table[(i, i+1)][lhs] = rhs[0]
                probs[(i, i+1)][lhs] = math.log(prob)

        # Fill in the tables for longer spans
        for span in range(2, n+1):
            for i in range(n-span+1):
                j = i + span
                for k in range(i+1, j):
                    for rule_list in self.grammar.lhs_to_rules.values():
                        for rule in rule_list:
                            A = rule[0]
                            rhs = rule[1]
                            prob = rule[2]
                            if len(rhs) == 2:
                                B, C = rhs
                                if B in table[(i, k)] and C in table[(k, j)]:
                                    new_prob = math.log(prob) + probs[(i, k)][B] + probs[(k, j)][C]
                                    if A not in probs[(i, j)] or new_prob > probs[(i, j)][A]:
                                        table[(i, j)][A] = ((B, i, k), (C, k, j))
                                        probs[(i, j)][A] = new_prob

        return table, probs

def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    if j - i == 1:
        output = (nt, chart[i, j][nt])
        return output

    out1 = get_tree(chart, chart[(i, j)][nt][0][1], chart[(i, j)][nt][0][2], chart[(i, j)][nt][0][0])
    out2 = get_tree(chart, chart[(i, j)][nt][1][1], chart[(i, j)][nt][1][2], chart[(i, j)][nt][1][0])
    return (nt, out1, out2)
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 

        # Part1 test
        print("Part1 test:")
        print("startsymbol:\n", grammar.startsymbol)
        print("lhs_to_rules['PP']:\n", grammar.lhs_to_rules['PP'])
        print("rhs_to_rules[('ABOUT','NP')]:\n", grammar.rhs_to_rules[('ABOUT','NP')])
        
        # Part2 test
        print("Part2 test:")
        parser = CkyParser(grammar)

        toks_1 =['flights', 'from','miami', 'to', 'cleveland','.'] 
        print(parser.is_in_language(toks_1))
        toks_2 =['miami', 'flights','cleveland', 'from', 'to','.']
        print(parser.is_in_language(toks_2))

        # Part3 test
        print("Part3 test:")
        toks = ['flights', 'from', 'miami', 'to', 'cleveland', '.']
        table, probs = parser.parse_with_backpointers(toks)
        print(check_table_format(table))
        print(check_probs_format(probs))

        # Part4 test
        print("Part4 test:")
        print(get_tree(table, 0, len(toks), grammar.startsymbol))
        
