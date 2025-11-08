import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull, QhullError
from scipy.optimize import linprog
from cdd import polyhedron_from_matrix, RepType, matrix_from_array, copy_generators

# ---------- Game representation ----------
def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0: return v
    return v / n

def _outward_normal(p, a, b):
    """
    Edge from a -> b; outward normal pointing 'outside' relative to polygon orientation.
    We assume vertices are in CCW order; outward normal is the CW perpendicular.
    """
    e = b - a
    n = np.array([ e[1], -e[0] ])  # CW perpendicular
    n = _unit(n)
    # If p is the polygon centroid, flip to ensure outward: dot(n, mid - p) > 0
    mid = 0.5*(a+b)
    if np.dot(n, mid - p) < 0: n = -n
    return n

def _safe_hull(points):
    pts = np.asarray(points, dtype=float)
    pts = np.unique(np.round(pts, 12), axis=0)
    if len(pts) <= 2:
        return pts
    try:
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except QhullError:
        return pts

def support_point(alpha, extW, game, delta, p, gbar):
    """Query LP oracle in direction alpha; return (v*, h=alpha·v*, success)."""
    alpha = _unit(alpha)
    ok, v, q, y = _solve_blockLP_in_direction(alpha, extW, game, delta, p, gbar)
    if not ok or not np.isfinite(v).all():
        return None, None, False
    return v, float(np.dot(alpha, v)), True

def refine_B_once(extW, game, delta, p, gbar, eps=1e-4, A0=None, max_refine_rounds=50, plot=False):
    """
    Compute an eps-accurate hull for B(W) by adaptive support-function refinement.
    Returns hull vertices (CCW).
    """
    if len(extW) == 0:
        return np.zeros((0,2))

    # 1) seed directions
    if A0 is None:
        A0 = [
            np.array([1.0, 0.0]), np.array([0.0, 1.0]),
            np.array([-1.0, 0.0]), np.array([0.0, -1.0]),
            np.array([1.0, 1.0]), np.array([1.0, -1.0]),
            np.array([-1.0, 1.0]), np.array([-1.0, -1.0]),
        ]

    # 2) initial samples
    samples = []
    for a in A0:
        v, h, ok = support_point(a, extW, game, delta, p, gbar)
        if ok:
            samples.append(v)
    if not samples:
        return np.zeros((0,2))

    P = _safe_hull(samples)
    changed = True
    rounds = 0

    while changed and rounds < max_refine_rounds and len(P) >= 2:
        changed = False
        rounds += 1

        # centroid used to orient outward normals
        centroid = np.mean(P, axis=0)

        # iterate edges of current hull
        new_pts = []
        for i in range(len(P)):
            a = P[i]
            b = P[(i+1) % len(P)]

            # 3) outward normal at edge
            n = _outward_normal(centroid, a, b)

            # 4) query oracle at n
            v_star, h_star, ok = support_point(n, extW, game, delta, p, gbar)
            if not ok:
                continue

            # 5) compute current support on this edge: use endpoint a (equivalently b)
            gap = h_star - float(np.dot(n, a))

            if gap > eps:
                new_pts.append(v_star)
                changed = True

        if new_pts:
            P = _safe_hull(np.vstack([P, np.array(new_pts)]))

    # optional overlay plot vs extW
    if plot and len(P) > 0:
        plt.figure(figsize=(6,6))
        poly = np.vstack([P, P[0]])
        plt.plot(poly[:,1], poly[:,0], '-o', label='B(W) hull (refined)')
        extW_arr = np.array(extW)
        if len(extW_arr):
            try:
                if len(extW_arr) > 2:
                    prev_hull = ConvexHull(extW_arr)
                    prev_poly = extW_arr[prev_hull.vertices]
                else:
                    prev_poly = extW_arr
            except QhullError:
                prev_poly = extW_arr
            if prev_poly.shape[0] == 1:
                plt.scatter(prev_poly[:,1], prev_poly[:,0], c='r', marker='x', label='W input (pt)')
            else:
                prev_poly = np.vstack([prev_poly, prev_poly[0]])
                plt.plot(prev_poly[:,1], prev_poly[:,0], '--', color='r', linewidth=1, label='W input')
        plt.xlabel('Follower payoff')
        plt.ylabel('Leader payoff')
        plt.title(f'B(W) via Support-Function Refinement (δ={delta}, rounds={rounds})')
        plt.grid(True); plt.axis('equal'); plt.legend(); plt.show()

    return P

def _hausdorff_distance(P, Q):
    """Approximate Hausdorff distance between two convex polygons given by vertices (Nx2 arrays)."""
    if len(P) == 0 or len(Q) == 0:
        return np.inf
    P, Q = np.asarray(P), np.asarray(Q)
    dPQ = np.max([np.min(np.linalg.norm(P - q, axis=1)) for q in Q])
    dQP = np.max([np.min(np.linalg.norm(Q - p, axis=1)) for p in P])
    return max(dPQ, dQP)


def iterate_B_with_refinement(
    W0, game, delta, p, gbar,
    eps=1e-4, max_refine_rounds=50,
    max_outer_iters=50, tol_outer=1e-4,
    plot_each=False,
    record_history=True
):
    """
    Iteration of B(W) = B_operator_with_refinement(W), using Hausdorff distance for convergence.
    Stops when ||W_{k+1} - W_k||_H < tol_outer.
    """
    Wk = _safe_hull(W0)
    history = [Wk.copy()] if record_history else None

    for it in range(max_outer_iters):
        print(f"[Outer] Iteration {it+1}")

        Wk1 = refine_B_once(Wk, game, delta, p, gbar,
                            eps=eps, max_refine_rounds=max_refine_rounds,
                            plot=False)

        d = _hausdorff_distance(Wk, Wk1)
        print(f"  Hausdorff Δ(W_{it}, W_{it+1}) ≈ {d:.3e}, |W|={len(Wk)} → {len(Wk1)}")

        if record_history:
            history.append(Wk1.copy())

        if d < tol_outer:
            print(f"Converged in {it+1} iterations (Hausdorff tol={tol_outer}).")
            break

        Wk = Wk1

    if record_history:
        return Wk, history
    return Wk

def plot_history_overlays(history, title="Convergence of $W_{k+1}=B(W_k)$",
                          base=None, show_points=False):
    """
    history: list of (N_k x 2) arrays (CCW), from iterate_B_with_refinement
    base:    optional initial extW or any reference set to overlay (drawn in red dashed)
    """
    plt.figure(figsize=(6.5,6.5))
    K = len(history)
    colors = cm.viridis(np.linspace(0.15, 0.95, K))  # light→dark

    for k, poly in enumerate(history):
        if len(poly)==0: continue
        poly_closed = np.vstack([poly, poly[0]])
        alpha = 0.25 + 0.5*(k/(K-1) if K>1 else 1.0)  # fade in
        plt.plot(poly_closed[:,1], poly_closed[:,0], '-', lw=2,
                 color=colors[k], alpha=alpha, label=(f"W_{k}" if k in (0, K-1) else None))
        if show_points:
            plt.scatter(poly[:,1], poly[:,0], s=12, color=colors[k], alpha=alpha)

    if base is not None and len(base):
        base = _safe_hull(base)
        base_closed = np.vstack([base, base[0]])
        plt.plot(base_closed[:,1], base_closed[:,0], '--', color='red', lw=1.5, label='Initial $W_0$')

    plt.xlabel('Follower payoff')
    plt.ylabel('Leader payoff')
    plt.title(title)
    plt.axis('equal'); plt.grid(True)
    plt.legend()
    plt.show()

def _solve_blockLP_in_direction(alpha, extW, game, delta, p, gbar):
    """
    Solve: maximize alpha · v over (q, y) subject to:
      - q >= 0, sum_t q_t = 1
      - y_{t,k} >= 0, sum_k y_{t,k} = q_t  (linking)
      - follower IC constraints (linear in q and y)
    where v = sum_t (1-delta) g(t) q_t + delta sum_{t,k} w^k y_{t,k}.

    Returns (success, v_opt, q_opt, y_opt).
    """
    T = game.terminals
    m = len(T)
    K = len(extW)
    w = np.array(extW, dtype=float)           # shape (K,2)
    g = np.array([game.g[t] for t in T])      # shape (m,2)
    g1 = g[:, 1]                               # follower component

    # Threat for deviation (as in your code): min follower continuation in extW (fallback to p)
    p_threat = np.min(w[:,1]) if K > 0 else p

    # Variable layout: x = [ q (m) | y (m*K) ]
    def y_index(t_idx, k_idx):
        return m + t_idx*K + k_idx

    nvars = m + m*K

    # --- Objective: maximize alpha · v  <=>  minimize c^T x with c = -alpha · v(x)
    # v(x) = sum_t (1-delta) g[t] q_t + delta * sum_{t,k} w[k] y_{t,k}
    # => c_q[t] = - alpha · ((1-delta) g[t])
    #    c_y[t,k] = - alpha · (delta * w[k])
    c = np.zeros(nvars, dtype=float)
    for t_idx in range(m):
        c[t_idx] = - np.dot(alpha, (1.0 - delta) * g[t_idx])
    for t_idx in range(m):
        for k_idx in range(K):
            c[y_index(t_idx, k_idx)] = - np.dot(alpha, delta * w[k_idx])

    # --- Equality constraints
    Aeq_rows = []
    beq = []

    # (1) Sum q = 1
    row = np.zeros(nvars, dtype=float)
    row[:m] = 1.0
    Aeq_rows.append(row)
    beq.append(1.0)

    # (2) For each t: sum_k y_{t,k} - q_t = 0
    for t_idx in range(m):
        row = np.zeros(nvars, dtype=float)
        for k_idx in range(K):
            row[y_index(t_idx, k_idx)] = 1.0
        row[t_idx] = -1.0
        Aeq_rows.append(row)
        beq.append(0.0)

    A_eq = np.vstack(Aeq_rows)
    b_eq = np.array(beq, dtype=float)

    # --- Inequality constraints (A_ub x <= b_ub)
    Aub_rows = []
    bub = []

    # Follower ICs:
    # For each follower node fn, for each pair (follow, dev), define:
    # LHS = sum_{t in reach(follow)} [ ((1-delta)*g1[t] - v_dev) q_t + delta * sum_k w[k]_1 * y_{t,k} ] >= 0
    # Convert to -LHS <= 0
    for fn in game.follower_nodes:
        children = game.dev_options[fn]
        for child_follow in children:
            reach = set(game.terminals_from_node[child_follow])
            if not reach:
                continue
            # Precompute v_dev for each possible dev
            for child_dev in children:
                if child_dev == child_follow:
                    continue
                v_dev = (1.0 - delta) * gbar[(fn, child_dev)] + delta * p_threat
                row = np.zeros(nvars, dtype=float)
                for t_idx, tname in enumerate(T):
                    if tname in reach:
                        # q part
                        row[t_idx] -= ((1.0 - delta) * g1[t_idx] - v_dev)
                        # y part: follower component of w^k
                        for k_idx in range(K):
                            row[y_index(t_idx, k_idx)] -= (delta * w[k_idx, 1])
                Aub_rows.append(row)
                bub.append(0.0)

    A_ub = np.vstack(Aub_rows) if Aub_rows else np.zeros((0, nvars))
    b_ub = np.array(bub, dtype=float) if bub else np.zeros((0,))

    # --- Bounds: q >= 0, y >= 0
    bounds = [(0.0, None)] * nvars

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res.success:
        return False, None, None, None

    x = res.x
    q_opt = x[:m]
    y_opt = x[m:].reshape(m, K)

    # Recover optimal payoff v
    v = np.zeros(2)
    v += np.sum(((1.0 - delta) * g.T * q_opt).T, axis=0)      # sum_t (1-delta) g[t] q_t
    # You can write clearer:
    v += delta * np.sum((y_opt @ w), axis=0)                  # delta * sum_{t,k} y_{t,k} w[k]

    return True, v, q_opt, y_opt

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

# ---------- LP solver and polygon extraction ----------
def exact_E_polytope(game, u_map, delta, p, gbar, min_follower_continuation=None):
    """
    Compute the exact polytope E(Q(u),u) using pycddlib.
    Includes:
      - IC inequalities
      - nonnegativity
      - equality sum(q) = 1
    Returns: vertices of the convex hull in payoff space.
    
    min_follower_continuation: minimum follower payoff among continuation values in extW
                                (used instead of p in IC constraints if provided)
    """
    T = game.terminals
    m = len(T)
    g1 = np.array([game.g[t][1] for t in T])
    u1 = np.array([u_map[t][1] for t in T])
    coef_ic = (1 - delta) * g1 + delta * u1
    
    # Use min_follower_continuation instead of p if provided
    p_threat = min_follower_continuation if min_follower_continuation is not None else p

    # --- Build inequalities b - A q >= 0  in CDD's format [b, A] ---
    # Note: CDD format is b + A'q >= 0, so we use A' = -A for Aq >= b
    hrep_rows = []

    # IC constraints: For each child, the conditional expected payoff from following the 
    # recommendation to go to that child should be at least as good as deviating to any other child.
    # For child c: E[payoff | go to c] >= payoff from deviating to c'
    # E[payoff | go to c] = sum_{t in reachable(c)} (q_t / sum_{t' in reachable(c)} q_t') * coef_ic[t]
    # This is linear in q, so we can rewrite as:
    # sum_{t in reachable(c)} q_t * coef_ic[t] >= v_dev(c') * sum_{t' in reachable(c)} q_t'
    # Rearranging: sum_{t in reachable(c)} q_t * (coef_ic[t] - v_dev(c')) >= 0
    for fn in game.follower_nodes:
        children = game.dev_options[fn]
        # For each pair of children (c, c') where c is followed and c' is the deviation
        for child_follow in children:
            reachable_follow = game.terminals_from_node[child_follow]
            mask_follow = np.array([1 if t in reachable_follow else 0 for t in T])
            for child_dev in children:
                if child_follow == child_dev:
                    continue  # no need to compare a child to itself
                v_dev = (1 - delta) * gbar[(fn, child_dev)] + delta * p_threat
                # IC: sum_{t in reachable(child_follow)} q_t * coef_ic[t] >= v_dev * sum_{t in reachable(child_follow)} q_t
                # Rewrite: sum_{t in reachable(child_follow)} q_t * (coef_ic[t] - v_dev) >= 0
                ic_vector = mask_follow * (coef_ic - v_dev)
                row = np.hstack([0.0, ic_vector])  # 0 >= -ic_vector @ q, or ic_vector @ q >= 0
                hrep_rows.append(row)

    # Nonnegativity: q_i >= 0.  A is identity, b is 0.
    # cdd row is [0, e_i] where e_i is the standard basis vector
    for i in range(m):
        row = np.zeros(m + 1)
        row[i+1] = 1.0
        hrep_rows.append(row)

    # --- FIX: Represent sum(q) = 1 as two inequalities ---
    # 1. sum(q) <= 1  ->  1 - sum(q) >= 0. cdd row is [1, -1, -1, ...]
    ones = -np.ones(m)
    hrep_rows.append(np.hstack([1.0, ones]))
    # 2. sum(q) >= 1  -> -1 + sum(q) >= 0. cdd row is [-1, 1, 1, ...]
    hrep_rows.append(np.hstack([-1.0, -ones]))

    hrep_matrix = np.array(hrep_rows, dtype=float)

    # --- Build cddlib matrix from the combined H-representation ---
    mat = matrix_from_array(hrep_matrix)
    mat.rep_type = RepType.INEQUALITY

    # Construct the polyhedron and get its vertices (V-representation)
    poly = polyhedron_from_matrix(mat)
    gens = copy_generators(poly)  # get generators (vertices and rays)
    

    # Collect all vertex q vectors
    vertices_q = []
    for g in gens.array:
        if g[0] == 1: # g[0] is 1 for a vertex, 0 for a ray
            qv = np.array(g[1:], dtype=float)
            qv[np.abs(qv) < 1e-12] = 0.0  # numerical cleanup
            vertices_q.append(qv)

    if len(vertices_q) == 0:
        return None

    # Compute E(q,u) = M q
    g_mat = np.array([game.g[t] for t in T], dtype=float)  # shape (m,2)
    u_mat = np.array([u_map[t] for t in T], dtype=float)
    M = (1 - delta) * g_mat.T + delta * u_mat.T  # shape (2,m)

    E_vertices = [M @ qv for qv in vertices_q]
    E_vertices = np.array(E_vertices)

    # Convex hull (to remove redundant vertices)
    if E_vertices.shape[0] > 2:
        hull = ConvexHull(E_vertices)
        return E_vertices[hull.vertices]
    else:
        return E_vertices

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

W0 = np.array(extW, dtype=float)              # your initial set
W_star, hist = iterate_B_with_refinement(
    W0, game, delta, p_val, gbar_map,
    eps=1e-8, max_refine_rounds=60,
    max_outer_iters=30, tol_outer=1e-8,
    plot_each=False, record_history=True
)

plot_history_overlays(hist, base=W0, show_points=False)
