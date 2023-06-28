"""
This observer is based on the work of Li, Jianan et al. in their 2023 paper on Tree-Dimensional Bearing-Only Target Following.
"""

import numpy as np

class PseudoLinearKF:
    def __init__(self, ts, xi, first_measurement) -> None:
        self.P = np.diag([5**2, 5**2, 5**2, 5**2])
        self.xhat = np.array([[100*first_measurement.item(0), 100.*first_measurement.item(1), 10., 10.]]).T
        self.A = np.eye(4)
        self.A[0:2, 2:] = ts * np.eye(2)
        self.B = np.zeros((4, 2))
        self.B[0:2] = 1/2. * ts * np.eye(2)
        self.B[2:] = ts * np.eye(2)

        self.Q = np.diag([0.01, 0.01])
        self.R = np.diag([0.001, 0.001])

        self.xi_prev = xi

    def update(self, xi, unit_vec):
        self.xhat = self.A @ (self.xhat + self.xi_prev) - xi
        self.P = self.A @ self.P @ self.A.T + self.B @ self.Q @ self.B.T
        self.xi_prev = xi

        # measurement update
        H  = np.zeros((2,4))
        Proj = np.eye(2) - unit_vec @ unit_vec.T
        H[:, 0:2]= Proj
        V = np.linalg.norm(self.xhat[0:2]) * Proj
        K = self.P @ H.T @ np.linalg.pinv(H @ self.P @ H.T + V @ self.R @ V.T)

        self.xhat -= K @ H @ self.xhat
        self.P = (np.eye(4) - K @ H) @ self.P

