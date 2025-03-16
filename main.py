from GameNode import GameNode

# define a game
def game_structure():
    # only need to set max_depth & discount factor for the root node if the game is iterated

    # # original game (not iterated)
    # root = GameNode(player=1)

    # iterated game
    root = GameNode(player=1, discount_factor=0.9, max_depth=50) 
    
    # create the rest of the game tree
    c1 = GameNode(player=0)
    c2 = GameNode(player=0)
    l1 = GameNode(player=-1, payoff=(4, 4))
    l2 = GameNode(player=-1, payoff=(10, 0))
    l3 = GameNode(player=-1, payoff=(0, 10))
    l4 = GameNode(player=-1, payoff=(9, 9))

    # connect the nodes
    root.add_child(c1)
    root.add_child(c2)
    c1.add_child(l1)
    c1.add_child(l2)
    c2.add_child(l3)
    c2.add_child(l4)

    # set the root node for all nodes
    root.set_root([root, c1, c2, l1, l2, l3, l4])
    return root

def main():
    game = game_structure()
    game.draw_EPF()

if __name__ == "__main__":
    main()


