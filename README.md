# EPF Solver for Two Players General Sum Repeated Games

## 1. Define a Game
1. **Create a root node** (and indicate the maximum depth and discount factor if the game is iterated).
2. **Create all remaining nodes** for the game tree. For all leaf nodes, indicate their payoff in the form **(follower payoff, leader payoff)**.
3. **Connect the game nodes** by calling `add_child` on parents.
4. **Set ALL game nodes** (including the root itself) to be the root by calling the function `set_root` on the root.

## 2. Visualize the Game
Call the `draw_EPF` method on the root node and you'll see the following window:

![EPF Visualization](sample_window.png)

## 3. Interact with the Slider
Interact with the slider to see the EPF of the game at different depths. The slider sometimes takes more than 10 seconds to load. If it doesn't respond for a long time, try switching to another screen and then coming back. This trick works all the time, at least on my side.

## 4. Change Viewing Options
Click the button below to change the viewing option (dynamic/static axes).

## Some Issues
1. **Naive Dynamic Programming**: As I used a very naive dynamic programming approach to compute the EPF, my code cannot handle super deep game trees now. (Will fix later.) Currently, for this prisoners' dilemma, I can set the `max_depth` to up to 150.
2. **Code Correctness**: I have not tested the correctness of my code. I have only used the prisoners' dilemma to run the program. Will fix later.

## Update logs:
1. **25/03/2025**: I fixed the bug that a point will be duplicated when it lies on the threshold. I also correct the algorithm so that the payoff of leaf nodes is calculated correctly.