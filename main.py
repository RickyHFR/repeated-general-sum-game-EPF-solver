import sys
sys.setrecursionlimit(3000)

from GameNode import GameNode

LEADER = 0
FOLLOWER = 1
LEAF = -1

# define a game
def game_structure():
    # only need to set max_depth & discount factor for the root node if the game is iterated

    # # original game (not iterated)
    # root = GameNode(player=1)

    # iterated game
    # root = GameNode(player=FOLLOWER, discount_factor=0.6, max_depth=200) 
    
    # # chicken game
    # # c1 = GameNode(player=LEADER)
    # # c2 = GameNode(player=LEADER)
    # # l1 = GameNode(player=LEAF, payoff=(9, 9))
    # # l2 = GameNode(player=LEAF, payoff=(8, 10))
    # # l3 = GameNode(player=LEAF, payoff=(10, 8))
    # # l4 = GameNode(player=LEAF, payoff=(0, 0))

    # prisoner's dilemma
    # c1 = GameNode(player=LEADER)
    # c2 = GameNode(player=LEADER)
    # l1 = GameNode(player=LEAF, payoff=(0, 11))
    # l2 = GameNode(player=LEAF, payoff=(2, 0))
    # l3 = GameNode(player=LEAF, payoff=(1, 1))
    # l4 = GameNode(player=LEAF, payoff=(1.5, 5))

    # # connect the nodes
    # root.add_child(c1)
    # root.add_child(c2)
    # c1.add_child(l1)
    # c1.add_child(l2)
    # c2.add_child(l3)
    # c2.add_child(l4)

    # set the root node for all nodes
    # root.set_root([root, c1, c2, l1, l2, l3, l4])
    # return root

    # slightly more complex game
    root = GameNode(player=LEADER, discount_factor=0.6, max_depth=100)
    n1 = GameNode(player=FOLLOWER)
    n2 = GameNode(player=FOLLOWER)
    n3 = GameNode(player=LEADER)
    n4 = GameNode(player=LEADER)
    n5 = GameNode(player=FOLLOWER)
    t1 = GameNode(player=LEAF, payoff=(1, 4))
    t2 = GameNode(player=LEAF, payoff=(4, 3))
    t3 = GameNode(player=LEAF, payoff=(5, 2))
    t4 = GameNode(player=LEAF, payoff=(0, 1))
    t5 = GameNode(player=LEAF, payoff=(3, 2))
    t6 = GameNode(player=LEAF, payoff=(6, 0))
    t7 = GameNode(player=LEAF, payoff=(2, 3))
    root.add_child(n1)
    root.add_child(n2)
    n1.add_child(n3)
    n1.add_child(t1)
    n2.add_child(n4)
    n2.add_child(t7)
    n3.add_child(t2)
    n3.add_child(n5)
    n4.add_child(t5)
    n4.add_child(t6)
    n5.add_child(t3)
    n5.add_child(t4)
    root.set_root([root, n1, n2, n3, n4, n5, t1, t2, t3, t4, t5, t6, t7])
    return root

def main():
    game = game_structure()
    game.draw_EPF()

if __name__ == "__main__":
    main()


