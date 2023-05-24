"""
    EKF for a target moving with constant velocity
"""

import numpy as np
from numpy import sin, cos
from msg.twoDYawState import TwoDYawState
from msg.bearing_msg import BearingMsg

class TargetEKF:
    def __init__(self, ts) -> None:
        self.Q = 0.707*np.diag([0.001, 0.001, 0.01, 0.001])
        self.R = np.diag([0.001**2])
        self.xhat = np.array([[0., 0., 30., 0.]]).T
        self.P = np.diag([10**2, 10**2, 5**2, np.pi**2])
        self.N = 20
        self.Ts = ts
        self.Tp = ts/self.N

    def update(self, measurement:BearingMsg, state:TwoDYawState, input):
        self.propagate_model(measurement, state, input)
        self.measurement_update(measurement, state)

    def propagate_model(self, measurement, state, input):
        for i in range(self.N):
            # propagate model
            self.xhat += self.Tp*self._f(self.xhat, measurement, state, input)
            self.xhat[3,0]=wrap(self.xhat[3,0])
            # get values for computing jacobian
            uavpos = state.getPos()
            vi = self.xhat.item(2)
            psii = self.xhat.item(3)
            # compute the jacobian
            A = np.array([[0.,0.,sin(psii), vi*cos(psii)],
                          [0.,0.,cos(psii), -vi*sin(psii)],
                          [0.,0.,0.,0.],
                          [0.,0.,0.,0.]])
            # convert to discrete time model
            A_d = np.identity(4) + A*self.Tp + A@A*self.Tp**2
            # update P with discrete time model
            self.P = A_d @self.P @ A_d.T + self.Tp**2 * self.Q

    def measurement_update(self, measurement, state):
        xi = self.xhat.item(0)
        yi = self.xhat.item(1)
        uavpos = state.getPos()
        xo = uavpos.item(0)
        yo = uavpos.item(1)
        psio = state.yaw
        h = np.array([[np.arctan2((xi-xo),(yi-yo))-psio]])
        C = np.array([[(-yi+yo)/((xi-xo)**2+(yi-yo)**2), (xi-xo)/((xi-xo)**2+(yi-yo**2)), 0.,0.]])
        y = np.array([[measurement.bearing]])
        S_inv = np.linalg.inv(self.R + C @ self.P @ C.T)
        L = self.P @ C.T @ S_inv
        self.P = (np.identity(5) - L @ C) @ self.P @ (np.identity(4)-L @ C).T + L @ self.R @ L.T
        self.xhat += L @ (y-h)

    def _f(self, x, measurement, state, input):
        # get values needed for the calculation
        vi = self.xhat.item(2)
        psii = self.xhat.item(3)
        # calculate xdot
        xdot = np.array([[vi*sin(psii), vi*cos(psii), 0., 0.]]).T
        return xdot
    
def wrap(angle):
    while angle > np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle