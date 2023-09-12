import numpy as np
import numpy.linalg as la
import math
from scipy.optimize import minimize, NonlinearConstraint
from geomdl import BSpline
from geomdl import utilities

class PathPlanner:

    def __init__(self, goal_pos, time_forward=5., max_velocity=23., ts=0.2) -> None:
        self.timestep_max_dist = max_velocity*ts
        self.num_control_points = int(time_forward/ts)
        self.ts = ts
        self.goal_pos = goal_pos
        self.old_path = np.zeros(self.num_control_points*2)
        self.probability_threshold = 0.0001
        pass

    def update(self, own_pos, intruder_pdfs) -> BSpline.Curve: # expect a nested list of intruder pdfs for each timestep into the future
        start_point=(0,0) # TODO: change to incorporate new initial position

        # setup the intruder avoidance constraint
        def calc_probability_collision(x):
            tmp = []

            for i in range(0, len(x), 2):
                out = 0
                for j in range(len(intruder_pdfs)):
                    if intruder_pdfs[j][i//2] is not None:
                        out += intruder_pdfs[j][i//2]([x[i],x[i+1]])
                tmp.append(out)
            return tmp
        avoidance_constraint = NonlinearConstraint(calc_probability_collision, 0.0, self.probability_threshold)

        # setup the maximum velocity constraint
        def calc_dx(x): # calculate the distance between the control points
            con = []
            con.append(math.dist((start_point[0],start_point[1]),(x[0],x[1])))
            for i in range(0, len(x)-2, 2):
                con.append(math.dist((x[i],x[i+1]),(x[i+2],x[i+3])))
            return con
        max_velocity_constraint = NonlinearConstraint(calc_dx, 0., self.timestep_max_dist)

        # drive us toward the objective function
        def objective_function(x):
            res = math.dist((x[-2],x[-1]), self.goal_pos)
            return res
        initial_x = self.old_path # TODO: transition from old path to current guess better (skip first point, add another endpoint)
        bounds = [(-10000,100) for i in range(len(initial_x))]
        res = minimize(objective_function, initial_x, method='SLSQP', constraints=[max_velocity_constraint, avoidance_constraint])
        cp = [[start_point[0], start_point[1]]]
        for i in range(0, len(res.x), 2):
            cp.append([res.x[i], res.x[i+1]])
        self.old_path = res.x

        curve = BSpline.Curve()
        curve.degree=4
        curve.ctrlpts = cp
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        curve.delta = 0.01
        return curve, cp
