"""
    EKF for a target moving with constant velocity, parameterized by inverse depth
"""

import numpy as np
from numpy import sin, cos
from msg.twoDYawState import TwoDYawState
from msg.bearing_msg import BearingMsg

class InverseDepthEKF:
    def __init__(self, initial_bearing, initial_yaw, ts) -> None:
        self.Q = 0.005*np.diag([0.1, 0.01, 0.01, 0.01, 0.1])
        self.R = np.diag([0.001**2, 0.01**2])
        self.xhat = np.array([[initial_bearing, 1/(20*39.), 15., np.pi/2, initial_yaw]]).T
        self.P = np.diag([0.01, 5**2, 5**2, np.pi**2, 0.01])
        self.N = 10
        self.Ts = ts
        self.Tp = ts/self.N

    def update(self, measurement:BearingMsg, state:TwoDYawState, input):
        self.propagate_model(measurement, state, input)
        self.measurement_update(measurement, state)

    def propagate_model(self, measurement, state, input):
        for i in range(self.N):
            # propagate model
            self.xhat += self.Tp*self._f(self.xhat, measurement, state, input)
            # self.xhat[1,0] = saturate(self.xhat[1,0], 0, 100000)
            self.xhat[2,0] = saturate(self.xhat[2,0], 0., 1000.)
            self.xhat[3,0] = wrap(self.xhat[3,0])
            self.xhat[4,0] = wrap(self.xhat[4,0])
            # get values for computing jacobian
            eta = self.xhat.item(0)
            rho = self.xhat.item(1)
            vi = self.xhat.item(2)
            psii = self.xhat.item(3)
            psi = self.xhat.item(4)
            vo = state.vel
            # compute the jacobian
            A = np.array([[rho*vo*cos(eta)-rho*vi*cos(eta-psii+psi), vo*sin(eta)-vi*sin(eta-psii+psi), -rho*sin(eta-psii+psi), rho*vi*cos(eta-psii+psi), -rho*vi*cos(eta-psi+psi)],
                          [rho**2*(-vo*sin(eta)+vi*sin(eta-psi+psi)), 2*rho*(vo*cos(eta)-vi*cos(eta-psi+psi)),-rho**2*cos(eta-psii+psi), -rho**2*vi*sin(eta-psii+psi), rho**2*vi*sin(eta-psii+psi)],
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
        # self.xhat[1,0] = saturate(self.xhat[1,0], 0.01, 100000)
        self.xhat[2,0] = saturate(self.xhat[2,0], 0., 1000.)
        self.xhat[3,0] = wrap(self.xhat[3,0])

    def _f(self, x, measurement, state, input):
        # get values needed for the calculation
        eta = x.item(0)
        rho = x.item(1)
        vi = x.item(2)
        psii = x.item(3)
        psi = x.item(4)
        vo = state.vel 
        psid = input
        # calculate xdot
        xdot = np.array([[vo*rho*sin(eta)-vi*rho*sin(eta+psi-psii)-psid, (vo*cos(eta)-vi*cos(eta+psi-psii))*rho**2, 0., 0., psid]]).T
        return xdot
    
def wrap(angle):
    while angle > np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle
def saturate(value, minimum, maximum):
    return min(maximum, max(value, minimum))