import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class EPF:
    def __init__(self, knots: np.ndarray):
        self.knots = knots

    def left_truncate(self, threshold):
        print(threshold)
        print(self.knots)
        idx = np.searchsorted(self.knots[:, 0], threshold, side='left')
        print(idx)
        if idx == 0:
            return self
        elif idx == self.knots.shape[0]:
            return EPF(None)
        elif idx > 0 and idx < self.knots.shape[0]:
            if self.knots[idx, 0] == threshold:
                return EPF(self.knots[idx:])
            else:
                # linear interpolation
                x1, y1 = self.knots[idx - 1]
                x2, y2 = self.knots[idx]
                y = y1 + (y2 - y1) / (x2 - x1) * (threshold - x1)
                return EPF(np.concatenate((np.array([[threshold, y]]), self.knots[idx:]), axis=0))

    def find_concave_envelope(self, target_EPF: 'EPF'):
        if target_EPF is None or target_EPF.knots is None:
            return self
        if self.knots is None:
            return target_EPF
        points = np.concatenate((self.knots, target_EPF.knots), axis=0)
        if points.shape[0] < 3:
            sorted_points = points[np.argsort(points[:, 0])]
            return EPF(sorted_points)
        
        hull = ConvexHull(points, qhull_options='QJ')
        hull_points = points[hull.vertices]
        hull_points = hull_points[np.argsort(hull_points[:, 0])]
        upper_hull = []
        for p in hull_points:
            while len(upper_hull) >= 2:
                p1, p2 = upper_hull[-2], upper_hull[-1]
                if np.cross(p2 - p1, p - p2) < 0:
                    break
                upper_hull.pop()
            upper_hull.append(p)
        
        return EPF(np.array(upper_hull))
    
    def min_follower_payoff(self):
        return self.knots[0, 0]


class GameNode:
    def __init__(self, player: int, payoff:tuple[float, float]=None, is_root: bool=False):
        self.is_root = is_root
        self.player = player       
        self.payoff = payoff       
        self.parent = None        
        self.children = []        
        self.EPF = None
        self.grim_value = None

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.set_parent(self)

    def set_parent(self, parent_node):
        self.parent = parent_node
    
    def get_grim_value(self):
        if self.grim_value is not None:
            return self.grim_value
        if self.player == -1:
            self.grim_value = self.payoff[1]
        elif self.player == 0:
            self.grim_value = min([child.get_grim_value() for child in self.children])
        else:
            self.grim_value = max([child.get_grim_value() for child in self.children])
        return self.grim_value  
    
    def get_EPF(self):
        if self.EPF is not None:
            return self.EPF
        if self.player == -1:
            self.EPF = EPF(np.array([[self.payoff[0], self.payoff[1]]]))
        if self.player == 0:
            result_EPF = EPF(None)
            for child in self.children:
                result_EPF = result_EPF.find_concave_envelope(child.get_EPF())
            self.EPF = result_EPF
        if self.player == 1:
            result_EPF = EPF(None)
            for child in self.children:
                result_EPF = result_EPF.find_concave_envelope(child.get_EPF())
            result_EPF = result_EPF.left_truncate(self.get_grim_value())
            self.EPF = result_EPF
        return self.EPF
    
    def draw_EPF(self):
        EPF = self.get_EPF()
        print(EPF.knots)
        plt.plot(EPF.knots[:, 0], EPF.knots[:, 1])


# define a simple prisoner's dilemma game
def prisoners_dilemma():
    root = GameNode(player=1, is_root=True)
    c1 = GameNode(player=0)
    c2 = GameNode(player=0)
    l1 = GameNode(player=-1, payoff=(4, 4))
    l2 = GameNode(player=-1, payoff=(0, 10))
    l3 = GameNode(player=-1, payoff=(10, 0))
    l4 = GameNode(player=-1, payoff=(9, 9))
    root.add_child(c1)
    root.add_child(c2)
    c1.add_child(l1)
    c1.add_child(l2)
    c2.add_child(l3)
    c2.add_child(l4)

    return root

game = prisoners_dilemma()
# print(game.get_EPF().knots)
# print(game.get_grim_value())
game.draw_EPF()
plt.show()
