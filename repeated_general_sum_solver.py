from typing import Union
import numpy as np  
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from GameNode import GameNode

# define a simple prisoner's dilemma game
def prisoners_dilemma():
    root = GameNode(player=1, max_depth=200)
    c1 = GameNode(player=0)
    c2 = GameNode(player=0)
    l1 = GameNode(player=-1, payoff=(4, 4))
    l2 = GameNode(player=-1, payoff=(10, 0))
    l3 = GameNode(player=-1, payoff=(0, 10))
    l4 = GameNode(player=-1, payoff=(9, 9))
    root.add_child(c1)
    root.add_child(c2)
    c1.add_child(l1)
    c1.add_child(l2)
    c2.add_child(l3)
    c2.add_child(l4)
    root.set_root([root, c1, c2, l1, l2, l3, l4])
    return root

game = prisoners_dilemma()
game.draw_EPF()
