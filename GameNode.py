from typing import Union

from matplotlib import pyplot as plt
import numpy as np
from EPF import EPF

class GameNode:
    def __init__(self,
                 player: int,
                 payoff: Union[tuple[float, float], EPF] = None,
                 discount_factor: float=0.9,
                 max_depth: int=50):
        self.player = player       
        self.payoff = payoff       
        self.parent = None
        self.children = []        
        self.discount_factor = discount_factor
        self.max_depth = max_depth
        self.root = None
        self.grim_memoization = {}
        self.EPF_memoization = {}

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.set_parent(self)

    def set_parent(self, parent_node):
        self.parent = parent_node

    def set_root(self, nodes):
        for node in nodes:
            node.root = self

    def get_grim_value(self, curr_depth = 0):
        if self.root.grim_memoization.get((self, curr_depth)) is not None:
            return self.root.grim_memoization[(self, curr_depth)]
        else:
            if self.player == -1 and curr_depth == self.root.max_depth:
                self.root.grim_memoization[(self, curr_depth)] = self.payoff[0] * self.discount_factor ** curr_depth
            elif self.player == -1 and curr_depth < self.root.max_depth:
                self.root.grim_memoization[(self, curr_depth)] = self.payoff[0] * self.discount_factor ** curr_depth + self.root.get_grim_value(curr_depth + 1) * self.discount_factor ** (curr_depth + 1)
            elif self.player == 0:
                self.root.grim_memoization[(self, curr_depth)] = min([child.get_grim_value(curr_depth) for child in self.children])
            else:
                self.root.grim_memoization[(self, curr_depth)] = max([child.get_grim_value(curr_depth) for child in self.children])
        return self.root.grim_memoization[(self, curr_depth)] 

    def get_EPF(self, curr_depth = 0):
        if self.root.EPF_memoization.get((self, curr_depth)) is not None:
            return self.root.EPF_memoization[(self, curr_depth)]
        else:
            if self.player == -1 and curr_depth == self.root.max_depth:
                self.root.EPF_memoization[self, curr_depth] = EPF(np.array([[self.payoff[0], self.payoff[1]]])).scale_then_shift((0, 0), self.discount_factor ** curr_depth)
            elif self.player == -1 and curr_depth < self.root.max_depth:
                self.root.EPF_memoization[self, curr_depth] = self.root.get_EPF(curr_depth + 1).scale_then_shift(
                    (self.payoff[0] * self.discount_factor ** curr_depth, self.payoff[1] * self.discount_factor ** curr_depth),
                    self.discount_factor)
            if self.player == 0:
                result_EPF = EPF(None)
                for child in self.children:
                    result_EPF = result_EPF.find_concave_envelope(child.get_EPF(curr_depth))
                self.root.EPF_memoization[self, curr_depth] = result_EPF
            if self.player == 1:
                result_EPF = EPF(None)
                for child in self.children:
                    child_threshold = max([other_child.get_grim_value(curr_depth) for other_child in self.children if other_child != child])
                    result_EPF = result_EPF.find_concave_envelope(child.get_EPF(curr_depth))
                    result_EPF = result_EPF.left_truncate(child_threshold)
                self.root.EPF_memoization[self, curr_depth] = result_EPF
            return self.root.EPF_memoization[self, curr_depth]
    
    def draw_EPF(self):
        if self.root.EPF_memoization.get((self, 0)) is None:
            self.get_EPF()
        dict_EPF = self.root.EPF_memoization
        for i in range(self.root.max_depth + 1):
            EPF_i = dict_EPF[(self, i)]
            plt.plot(EPF_i.knots[:, 0], EPF_i.knots[:, 1])
        plt.show()