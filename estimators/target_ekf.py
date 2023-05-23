"""
    EKF for a target moving with constant velocity
"""

import numpy as np
from numpy import sin, cos
from msg.twoDYawState import TwoDYawState
from msg.bearing_msg import BearingMsg

class TargetEKF:
    def __init__(self, initial_bearing, initial_yaw, ts) -> None:
        self.Q = np.diag([0.1, 0.5, 0.1, 0.1, 0.1])
        self.R = np.diag([0.01**2, 0.01**2])
        self.xhat = np.array([[initial_bearing, 39., 15., np.pi/2, initial_yaw]]).T
        self.P = np.diag([0.1, 100., 1., 0.5, 0.01])
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
            # get values for computing jacobian
            eta = self.xhat.item(0)
            tau = self.xhat.item(1)
            vi = self.xhat.item(2)
            psii = self.xhat.item(3)
            psi = self.xhat.item(4)
            vo = state.vel
            # compute the jacobian
            A = np.array([[cos(eta)/tau-vi*cos(eta+psi-psii)/(tau*vo), -sin(eta)/tau**2+vi*sin(eta+psi-psii)/(tau**2*vo), -sin(eta+psi-psii)/(tau*vo), vi*cos(eta+psi-psii)/(tau*vo), -vi*cos(eta+psi-psii)/(tau*vo)],
                          [sin(eta)-sin(eta+psi-psii)*vi/vo, 0, cos(eta+psi-psii)/vo, sin(eta+psi-psii)*vi/vo, -vi*sin(eta+psi-psii)/vo],
                          [0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.],
                          [0.,0.,0.,0.,0.]])
            # convert to discrete time model
            A_d = np.identity(5) + A*self.Tp + A@A*self.Tp**2
            # update P with discrete time model
            self.P = A_d @self.P @ A_d.T + self.Tp**2 * self.Q

    def measurement_update(self, measurement, state):
        h = np.array([[self.xhat.item(0), self.xhat.item(4)]]).T
        C = np.array([[1., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 1.]])
        y = np.array([[measurement.bearing, measurement.yaw]]).T 
        S_inv = np.linalg.inv(self.R + C @ self.P @ C.T)
        L = self.P @ C.T @ S_inv
        self.P = (np.identity(5) - L @ C) @ self.P @ (np.identity(5)-L @ C).T + L @ self.R @ L.T
        self.xhat += L @ (y-h)

    def _f(self, x, measurement, state, input):
        # get values needed for the calculation
        eta = x.item(0)
        tau = x.item(1)
        vi = x.item(2)
        psii = x.item(3)
        psi = x.item(4)
        vo = state.vel 
        psid = input
        # calculate xdot
        xdot = np.array([[sin(eta)/tau-vi*sin(eta+psi-psii)/(vo*tau)-psid,-cos(eta)+vi/vo*cos(eta+psi-psii), 0., 0., psid]]).T
        return xdot