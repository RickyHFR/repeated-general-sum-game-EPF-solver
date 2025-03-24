import numpy as np
from scipy.spatial import ConvexHull

class EPF:
    def __init__(self, knots: np.ndarray):
        # Convert knots to high-precision if not None
        self.knots = np.array(knots, dtype=np.longdouble) if knots is not None else None

    def left_truncate(self, threshold):
        idx = np.searchsorted(self.knots[:, 0], np.longdouble(threshold), side='left')
        if idx == 0:
            return self
        elif idx == self.knots.shape[0]:
            return EPF(None)
        else:
            x1, y1 = self.knots[idx - 1]
            x2, y2 = self.knots[idx]
            y = y1 + (y2 - y1) / (x2 - x1) * (np.longdouble(threshold) - x1)
            return EPF(np.concatenate((np.array([[np.longdouble(threshold), y]], dtype=np.longdouble), self.knots[idx:]), axis=0))

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
    
    def scale_then_shift(self, shift: tuple[float, float], scale: float):
        if self.knots is None:
            return self
        shift_vector = np.array(shift, dtype=np.longdouble)
        return EPF(self.knots * np.longdouble(scale) + shift_vector)
    
    def min_follower_payoff(self):
        return self.knots[0, 0]