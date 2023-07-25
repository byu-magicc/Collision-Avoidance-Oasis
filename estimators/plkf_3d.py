"""
This observer is based on the work of Li, Jianan et al. in their 2023 paper on Tree-Dimensional Bearing-Only Target Following.
"""

import numpy as np

class PseudoLinearKF:
    def __init__(self, ts, xi, first_measurement) -> None:
        self.P = np.diag([5**2, 5**2, 5**2, 5**2, 5**2, 5**2])
        range_guess = 300. #guess of range to target
        velocity_guess = -30. # guess of relative velocity of target, that it is approaching us
        self.xhat = np.array([[range_guess*first_measurement.item(0), 
                               range_guess*first_measurement.item(1), 
                               range_guess*first_measurement.item(2), 
                               velocity_guess*first_measurement.item(0), 
                               velocity_guess*first_measurement.item(1), 
                               velocity_guess*first_measurement.item(2)]]).T
        self.A = np.eye(6)
        self.A[0:3, 3:] = ts * np.eye(3)
        self.B = np.zeros((6, 3))
        self.B[0:3] = 1/2. * ts * np.eye(3)
        self.B[3:] = ts * np.eye(3)

        self.Q = np.diag([1, 1, 1])
        self.R = np.diag([0.001, 0.001, 0.001])

        self.xi_prev = xi

    def update(self, xi, unit_vec):
        self.xhat = self.A @ (self.xhat + self.xi_prev) - xi
        self.P = self.A @ self.P @ self.A.T + self.B @ self.Q @ self.B.T
        self.xi_prev = xi

        # measurement update
        H  = np.zeros((3,6))
        Proj = np.eye(3) - unit_vec @ unit_vec.T
        H[:, 0:3]= Proj
        V = np.linalg.norm(self.xhat[0:3]) * Proj
        K = self.P @ H.T @ np.linalg.pinv(H @ self.P @ H.T + V @ self.R @ V.T)

        self.xhat -= K @ H @ self.xhat
        self.P = (np.eye(6) - K @ H) @ self.P

