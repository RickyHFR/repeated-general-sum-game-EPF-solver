import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class GameNode:
    def __init__(self, player: int, payoff:tuple[float, float]=None, is_root: bool=False):
        self.is_root = is_root       # True if this is the root node
        self.player = player       # whose turn or decision point
        self.payoff = payoff       # terminal payoff (None for non-terminal nodes)
        self.parent = None         # parent GameNode instance
        self.children = []         # list of child GameNode instances

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.set_parent(self)

    def set_parent(self, parent_node):
        self.parent = parent_node
    
    def get_player(self):
        return self.player
    
# define a simple prisoner's dilemma game
def prisoners_dilemma():
    root = GameNode(player=0, is_root=True)
    c1 = GameNode(player=1)
    c2 = GameNode(player=1)
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

    return root

# auxiliary function to conpute upper concave envelope of a set of points
def upper_concave_envelope(points):
    points = np.array(points)
    hull = ConvexHull(points)
    # Extract the hull points
    hull_points = points[hull.vertices]
    # Sort by x-coordinates to find the upper hull
    hull_points = hull_points[np.argsort(hull_points[:, 0])]
    upper_hull = []
    for p in hull_points:
        while len(upper_hull) >= 2:
            p1, p2 = upper_hull[-2], upper_hull[-1]
            # Check if the new point maintains concavity (right-turn test)
            if np.cross(p2 - p1, p - p2) < 0:
                break
            upper_hull.pop()
        upper_hull.append(p)
    
    return np.array(upper_hull)
    
    

