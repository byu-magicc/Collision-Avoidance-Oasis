"""
    Controller designed to make the derivative of bearing to each obstacle non-zero
"""

import numpy as np
from typing import List
from tools.dirty_derivative import DirtyDerivative
from msg.bearing_msg import BearingMsg
from tools.tools import hat, euler_to_rotation

class HelicalNavigationLaw:
    def __init__(self, Ts) -> None:
        self.Ts = Ts
        self.t = 0
        self.N = 1.
        self.c_z = 1.
        self.c_y = 1.


    def update(self, desired_direction:np.ndarray, vel:np.ndarray):
        ex = np.array([[1., 0., 0.]]).T
        eta = hat(ex) @ desired_direction
        theta_e = np.arccos(desired_direction.T @ ex)
        skew_e = hat(eta)
        R_eta = np.eye(3) + np.sin(theta_e) * skew_e + (1-np.cos(theta_e)) * skew_e@skew_e
        Rz = euler_to_rotation(psi=self.c_z*np.sin(self.t))
        Ry = euler_to_rotation(theta=self.c_y*np.cos(self.t))

        des_accel = hat(hat(self.N * vel) @ R_eta@Rz@Ry@ex) @ desired_direction

        self.t += self.Ts
        return des_accel
        

    def saturate(self, input):
        if input > self.upper_bound:
            return self.upper_bound
        elif input < self.lower_bound:
            return self.lower_bound
        return input
