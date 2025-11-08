import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import ConvexHull

# ---------- Game representation ----------
class Node:
    def __init__(self, name, player, children=None, payoff=None):
        self.name = name
        self.player = player  # 'L' for leader, 'F' for follower, 'T' for terminal
        self.children = children or []  # list of Node objects
        self.payoff = payoff  # tuple (g0,g1) if terminal, else None

class Game:
    def __init__(self, root, follower_nodes):
        self.root = root
        # collect terminals, map node names to nodes, and follower nodes list
        self.nodes = {}
        self._collect_nodes(root)
        self.T_nodes = [n for n in self.nodes.values() if n.player == 'T']
        self.terminals = [n.name for n in self.T_nodes]
        self.terminal_index = {t:i for i,t in enumerate(self.terminals)}
        self.g = {n.name: n.payoff for n in self.T_nodes}
        self.follower_nodes = follower_nodes  # list of follower node names
        # For each follower node, store the names of child nodes that represent deviations (successor nodes)
        self.dev_options = {fn: [c.name for c in self.nodes[fn].children] for fn in follower_nodes}
        self.terminals_from_node = {n: self._get_reachable_terminals(self.nodes[n]) for n in self.nodes}

    def _collect_nodes(self, node):
        self.nodes[node.name] = node
        for c in node.children:
            self._collect_nodes(c)

    def _get_reachable_terminals(self, start_node):
        """Helper to find all terminal nodes reachable from a given start_node."""
        reachable = set()
        nodes_to_visit = [start_node]
        visited = set()
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            if current_node.name in visited:
                continue
            visited.add(current_node.name)
            if current_node.player == 'T':
                reachable.add(current_node.name)
            else:
                for child in current_node.children:
                    nodes_to_visit.append(child)
        return list(reachable)

# ---------- Backward induction to compute punishment values ----------
def follower_value_from(node):
    """
    Compute the value for the follower when leader plays adversarially (leader minimizes follower payoff,
    follower chooses to maximize when it's follower's turn). This is standard backward induction for
    a zero-sum objective where follower payoff is the value.
    """
    if node.player == 'T':
        return node.payoff[1]  # follower payoff at terminal
    if node.player == 'F':
        # follower chooses child that maximizes follower payoff
        vals = [follower_value_from(c) for c in node.children]
        return max(vals)
    if node.player == 'L':
        # leader (adversary) chooses child that minimizes follower payoff
        vals = [follower_value_from(c) for c in node.children]
        return min(vals)
    raise ValueError("Unknown node type")

def compute_p_and_gbar(game):
    """
    Compute punishment p (value starting from root when leader punishes) and
    for each follower child node n' compute gbar1(n') defined as one-shot payoff starting at that node
    with leader punishing afterwards.
    """
    p = follower_value_from(game.root)
    # For each follower node, for each child, compute the follower payoff starting from that child
    gbar = {}
    for fn in game.follower_nodes:
        for child_name in game.dev_options[fn]:
            child_node = game.nodes[child_name]
            gbar[(fn,child_name)] = follower_value_from(child_node)
    return p, gbar

# ---------- Geometry helpers ----------
def _convex_hull_ccw(points, round_ndigits=12):
    """Return CCW hull vertices of a point set (2D)."""
    P = np.unique(np.round(np.asarray(points, dtype=float), round_ndigits), axis=0)
    if len(P) == 0:
        return np.zeros((0,2))
    if len(P) <= 2:
        return P
    hull = ConvexHull(P)
    return P[hull.vertices]

def _clip_polygon_w1_lower(poly, w1_min, round_ndigits=12):
    """
    Sutherland–Hodgman clipping of a convex polygon 'poly' by the half-plane w1 >= w1_min,
    where coordinates are (leader, follower) -> use index 1 for follower.
    Returns the clipped polygon vertices in CCW order (could be empty).
    """
    if len(poly) == 0:
        return np.zeros((0,2))
    out = []
    n = len(poly)
    def inside(pt):
        return pt[1] >= w1_min - 1e-12
    def intersect(p1, p2):
        # Intersect segment p1->p2 with line w1 = w1_min (on follower axis, index 1)
        y1, y2 = p1[1], p2[1]
        if abs(y2 - y1) < 1e-18:
            return None  # parallel; caller only uses when one inside, one outside
        t = (w1_min - y1) / (y2 - y1)
        t = max(0.0, min(1.0, t))
        return p1 + t * (p2 - p1)

    for i in range(n):
        cur = poly[i]
        nxt = poly[(i+1) % n]
        cur_in = inside(cur)
        nxt_in = inside(nxt)
        if cur_in and nxt_in:
            out.append(nxt)
        elif cur_in and not nxt_in:
            ip = intersect(cur, nxt)
            if ip is not None:
                out.append(ip)
        elif (not cur_in) and nxt_in:
            ip = intersect(cur, nxt)
            if ip is not None:
                out.append(ip)
            out.append(nxt)
        # else both out -> add nothing
    out = np.unique(np.round(np.asarray(out, dtype=float), round_ndigits), axis=0)
    if len(out) == 0:
        return np.zeros((0,2))
    if len(out) <= 2:
        return out
    return _convex_hull_ccw(out, round_ndigits=round_ndigits)

# ---------- Profile enumeration & path evaluation ----------

def _enumerate_pure_profiles(game):
    """
    Enumerate pure within-period plans a: for each non-terminal node pick exactly one child.
    Returns a list of dicts {node_name: chosen_child_name}.
    """
    decision_nodes = [n for n in game.nodes.values() if n.player in ('L','F')]
    choices_per_node = []
    node_names = []
    for nd in decision_nodes:
        node_names.append(nd.name)
        choices_per_node.append([c.name for c in nd.children])
    profiles = []
    for combo in itertools.product(*choices_per_node):
        prof = {node_names[i]: combo[i] for i in range(len(node_names))}
        profiles.append(prof)
    return profiles

def _follow_path_and_collect(game, profile):
    """
    Follow the unique path induced by 'profile' from root until terminal.
    Return (terminal_name, follower_nodes_on_path_in_order).
    """
    cur = game.root
    follower_path = []
    visited = set()
    while cur.player != 'T':
        if cur.name in visited:
            raise RuntimeError("Cycle detected in tree.")
        visited.add(cur.name)
        if cur.player == 'F':
            follower_path.append(cur.name)
        # choose child prescribed by profile
        if cur.name not in profile:
            raise KeyError(f"Profile missing choice for node {cur.name}")
        chosen_child = profile[cur.name]
        cur = game.nodes[chosen_child]
    return cur.name, follower_path

def _max_follower_one_shot_gain(game, profile, terminal_name, gbar_map):
    """
    h_F(a) = max over follower nodes on path, and over deviations at that node, of
             [gbar_dev - g1_on_path]
    where gbar_dev is the follower's one-shot payoff from deviating (then facing punishment),
    and g1_on_path is the follower's one-shot payoff on the recommended terminal.
    """
    g1_on_path = game.g[terminal_name][1]
    _, follower_nodes_on_path = None, None  # placeholder (we already have path in caller)
    # We recompute follower nodes on path here to keep interface simple:
    _, follower_nodes_on_path = _follow_path_and_collect(game, profile)

    gains = []
    for fn in follower_nodes_on_path:
        follow_child = profile[fn]
        for dev_child in game.dev_options[fn]:
            if dev_child == follow_child:
                continue
            gains.append(gbar_map[(fn, dev_child)] - g1_on_path)
    if not gains:
        return 0.0
    return max(0.0, max(gains))

# ---------- R-operator (no LP) ----------

def R_operator(extW, game, delta, p, gbar, plot=True):
    """
    One application of the R-like operator (no LPs).
    extW: list/array of 2D points (leader, follower) defining current continuation extreme set W.
    Steps:
      - Build convex W polygon (CCW).
      - For each pure profile a:
          * get its terminal t(a) and g(a)
          * hF(a) := max one-shot follower gain on the path
          * w1_min := p_threat + ((1-delta)/delta) * hF(a)
          * Q := W ∩ {w: w1 >= w1_min}
          * Map Q -> V_a := {(1-delta)g(a) + delta*w : w in Q}
      - Return convex hull of union over a of V_a.
    """
    if len(extW) == 0:
        print("  Input: 0 extreme points in W")
        print("  Output: 0 extreme points (empty R(W))")
        return np.zeros((0,2))

    # Build W polygon (CCW) and compute follower-min threat p_threat from W
    W_poly = _convex_hull_ccw(extW)
    p_threat = float(np.min(W_poly[:,1])) if len(W_poly) else p

    profiles = _enumerate_pure_profiles(game)

    all_pts = []
    used_profiles = 0
    for prof in profiles:
        tname, _ = _follow_path_and_collect(game, prof)
        g_vec = np.array(game.g[tname], dtype=float)  # (g0, g1) at terminal on the path
        hF = _max_follower_one_shot_gain(game, prof, tname, gbar)
        # IC half-plane: w1 >= p_threat + ((1-delta)/delta) * hF
        if delta <= 0.0:
            continue
        w1_min = p_threat + ((1.0 - delta) / delta) * hF

        Q = _clip_polygon_w1_lower(W_poly, w1_min)
        if len(Q) == 0:
            continue

        used_profiles += 1
        # Map Q into current payoffs
        Va = (1.0 - delta) * g_vec + delta * Q  # affine map of each vertex
        all_pts.append(Va)

    if not all_pts:
        print("  Output: 0 extreme points (no feasible profiles)")
        return np.zeros((0,2))

    all_pts = np.vstack(all_pts)
    hull_pts = _convex_hull_ccw(all_pts)
    print(f"  Profiles considered: {len(profiles)}, feasible: {used_profiles}")
    print(f"  Output: {len(hull_pts)} extreme points in R(W)")

    if plot and len(hull_pts) > 0:
        poly = np.vstack([hull_pts, hull_pts[0]])
        plt.figure(figsize=(6,6))
        plt.plot(poly[:,1], poly[:,0], '-o', label='R(W) hull (R-like)')
        Wp = np.vstack([W_poly, W_poly[0]])
        plt.plot(Wp[:,1], Wp[:,0], '--', label='W')
        plt.xlabel('Follower payoff')
        plt.ylabel('Leader payoff')
        plt.title(f'R(W) (delta={delta})')
        plt.grid(True); plt.axis('equal'); plt.legend(); plt.show()

    return hull_pts

def _hausdorff_distance(P, Q):
    """Approximate Hausdorff distance between two convex polygons given by vertices (Nx2 arrays)."""
    if len(P) == 0 or len(Q) == 0:
        return np.inf
    # directed distances: for each vertex in P, distance to nearest vertex in Q
    dPQ = np.max([np.min(np.linalg.norm(P - q, axis=1)) for q in Q])
    dQP = np.max([np.min(np.linalg.norm(Q - p, axis=1)) for p in P])
    return max(dPQ, dQP)

def iterate_R(extW, game, delta, p, gbar, max_iter=200, tol=1e-9, plot_every=0, keep_history=False):
    """
    Fixed-point iteration W_{k+1} = R(W_k), with Hausdorff-distance termination.
    Args:
        extW: initial polygon vertices (list of (leader, follower) payoffs)
        game, delta, p, gbar: as before
        max_iter: maximum iterations
        tol: termination threshold for Hausdorff distance
        plot_every: if >0, plot every 'plot_every' iterations
        keep_history: if True, store all iterates in a list

    Returns:
        W_final: converged polygon vertices
        (optionally) history: list of all iterates if keep_history=True
    """
    Wk = _convex_hull_ccw(extW)
    history = [Wk] if keep_history else None

    for it in range(1, max_iter + 1):
        # Compute next iteration
        plot_flag = (plot_every > 0 and it % plot_every == 0)
        Wnext = R_operator(Wk, game, delta, p, gbar, plot=plot_flag)

        # Compute Hausdorff-like distance
        dist = _hausdorff_distance(np.array(Wk), np.array(Wnext))
        print(f"Iter {it:3d}: |W_{it} - W_{it-1}| ≈ {dist:.3e}, vertices: {len(Wnext)}")

        if dist <= tol:
            print(f"Converged in {it} iterations (tol={tol}).")
            if keep_history:
                history.append(Wnext)
            break

        if keep_history:
            history.append(Wnext)

        Wk = Wnext

    if keep_history:
        return Wnext, history
    return Wnext

def plot_R_history(history, highlight_final=True, highlight_initial=True, title="Convergence of $W_{k+1}=R(W_k)$"):
    """
    Plot the entire convergence history of R-operator iteration.
    Args:
        history: list of 2D numpy arrays, each (n_i, 2), the iterates W_k
        highlight_final: draw the final fixed point in bold color
        highlight_initial: draw the first set in dashed line
        title: plot title
    """
    plt.figure(figsize=(6,6))
    n = len(history)
    for i, W in enumerate(history):
        poly = np.vstack([W, W[0]])
        alpha = 0.2 + 0.8 * (i / (n - 1))  # gradual opacity
        plt.plot(poly[:,1], poly[:,0], '-', color='C0', alpha=alpha, linewidth=1.5)

    if highlight_initial and len(history) > 0:
        W0 = np.vstack([history[0], history[0][0]])
        plt.plot(W0[:,1], W0[:,0], '--', color='orange', linewidth=1.5, label='$W_0$')

    if highlight_final and len(history) > 1:
        Wf = np.vstack([history[-1], history[-1][0]])
        plt.plot(Wf[:,1], Wf[:,0], '-', color='green', linewidth=2.5, label='$W^* = W_{k}$')

    plt.xlabel("Follower payoff")
    plt.ylabel("Leader payoff")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()


# ---------- Toy example and run ----------
# t1 = Node('t1', 'T', children=[], payoff=(11, 0))
# t2 = Node('t2', 'T', children=[], payoff=(0, 2))
# t3 = Node('t3', 'T', children=[], payoff=(1, 1))
# t4 = Node('t4', 'T', children=[], payoff=(5, 1.5))
# n1 = Node('n1', 'L', children=[t1, t2])
# n2 = Node('n2', 'L', children=[t3, t4])
# root = Node('root', 'F', children=[n1, n2])

# game = Game(root, follower_nodes=['root'])

t1 = Node('t1', 'T', children=[], payoff=(4, 1))
t2 = Node('t2', 'T', children=[], payoff=(3, 4))
t3 = Node('t3', 'T', children=[], payoff=(2, 5))
t4 = Node('t4', 'T', children=[], payoff=(1, 0))
t5 = Node('t5', 'T', children=[], payoff=(2, 3))
t6 = Node('t6', 'T', children=[], payoff=(0, 6))
t7 = Node('t7', 'T', children=[], payoff=(3, 2))
n4 = Node('n4', 'L', children=[t5, t6])
n5 = Node('n5', 'F', children=[t3, t4])
n3 = Node('n3', 'L', children=[t2, n5])
n1 = Node('n1', 'F', children=[n3, t1])
n2 = Node('n2', 'F', children=[n4, t7])
root = Node('root', 'L', children=[n1, n2])
game = Game(root, follower_nodes=['n1', 'n2', 'n5'])
# compute p and gbar
p_val, gbar_map = compute_p_and_gbar(game)
print("Computed punishment p =", p_val)
print("gbar map:", gbar_map)

# extW for toy example (rectangle corners)
# extW = [(11, 0), (0, 2), (1, 1), (5, 1.5)]
# extW = [(6.4, 1), (0, 2), (1, 1), (5, 1.5)]
extW = [(4, 1), (3, 4), (2, 5), (1, 0), (2, 3), (0, 6), (3, 2)]
delta = 0.6

W1 = R_operator(extW, game, delta, p_val, gbar_map, plot=True)
W_star, history = iterate_R(extW, game, delta, p_val, gbar_map,
                            max_iter=50, tol=1e-8, plot_every=0, keep_history=True)

plot_R_history(history, title="Convergence Path of R-operator Iteration")
print("Final vertices of R^*(W):")
print(W_star)