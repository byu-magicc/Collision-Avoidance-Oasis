"""
    EKF for a target moving with constant velocity
"""

import numpy as np
from numpy import sin, cos
from msg.twoDYawState import TwoDYawState
from msg.bearing_msg import BearingMsg

class TTCUnscentedEKF:
    def __init__(self, initial_bearing, initial_yaw, ts) -> None:
        self.Q = 0.005*np.diag([0.1, 0.01, 0.01, 0.01, 0.1])
        self.R = np.diag([0.001**2, 0.01**2])
        self.mean = np.array([[initial_bearing, 39., 15., np.pi/2, initial_yaw]]).T
        self.P = np.diag([0.01, 5**2, 5**2, np.pi**2, 0.01])
        self.Ts = ts
        ll = 1
        self.n = 5
        self.w0 = ll/(self.n+ll)
        self.wi = 1/(2*(self.n+ll))
        self.c = np.sqrt(self.n + ll)

    def update(self, measurement:BearingMsg, state:TwoDYawState, input):
        self.propagate_model(state, input)
        self.measurement_update(measurement, state)

    def update_point(self, xhat, state:TwoDYawState, input):
        x1 = self._f(xhat, state, input)
        x2 = self._f(xhat + self.Ts/2.*x1, state, input)
        x3 = self._f(xhat + self.Ts/2*x2, state, input)
        x4 = self._f(xhat + self.Ts*x3, state, input)

        xhat += self.Ts/6.*(x1+2*x2+2*x3+x4)
        return xhat

    def propagate_model(self, state, input):

        # generate sigma points
        self.sigma_points = [np.zeros((5,1)),np.zeros((5,1)),np.zeros((5,1)),np.zeros((5,1)),np.zeros((5,1)),np.zeros((5,1)),np.zeros((5,1)),np.zeros((5,1)),np.zeros((5,1)),np.zeros((5,1)),np.zeros((5,1))]
        sqrtP = np.linalg.cholesky(self.P)
        self.sigma_points[0] = self.mean

        for i in range(1,self.n+1): #TODO: double check indexing here
            self.sigma_points[i]        = self.mean + self.c * np.reshape(sqrtP[:,i-1],(self.n,1))
            self.sigma_points[i+self.n] = self.mean - self.c * np.reshape(sqrtP[:,i-1],(self.n,1))
        # propogate each point through the dynamics
        for i in range(len(self.sigma_points)):
            self.sigma_points[i] = self.update_point(self.sigma_points[i], state, input)
            # points[i][1,0] = saturate(points[i][1,0], 0.01, 100000)
            # points[i][2,0] = saturate(points[i][2,0], 0., 1000.)

        #calculate the new mean
        self.mean = self.w0*self.sigma_points[0]
        for i in range(1,len(self.sigma_points)):
            self.mean += self.wi*self.sigma_points[i]
        
        # calculate the new covariance
        self.P = self.w0 * (self.sigma_points[0]-self.mean)@(self.sigma_points[0]-self.mean).T
        for i in range(1, len(self.sigma_points)):
            self.P += self.wi*(self.sigma_points[i]-self.mean)@(self.sigma_points[i]-self.mean).T
        self.P += self.Q #TODO: Q is a function of mean?

        

    def measurement_update(self, measurement:BearingMsg, state):
        ys = []
        # push the sigma points through the measurement function
        for i in range(len(self.sigma_points)):
            ys.append(self._h(self.sigma_points[i]))

        # compute the measurement mean
        mu = self.w0 * ys[0]
        for i in range(1, len(self.sigma_points)):
            mu += self.wi * ys[i]
        
        # compute the predicted measurement covariance
        S = self.w0 *(ys[0]-mu)@(ys[0]-mu).T
        for i in range(1, len(self.sigma_points)):
            S += self.wi*(ys[i]-mu)@(ys[i]-mu).T
        S += self.R #TODO: R is a function of the mean?

        # compute the predicted cross-covariance of state and measurement
        C = self.w0 * (self.sigma_points[0] - self.mean) @ (ys[0]-mu).T
        for i in range(1, len(self.sigma_points)):
            C += self.wi * (self.sigma_points[i]-self.mean)@(ys[i]-mu).T

        # compute the filter gain
        K = C@np.linalg.inv(S) #TODO: correct syntax for python

        # compute the filtered state mean
        y = np.array([[measurement.bearing, measurement.yaw]]).T
        self.mean += K@(y-mu)

        self.P -= K@S@K.T

    def _f(self, x, state, input):
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
    def _h(self, x):
        return np.array([[x.item(0), x.item(4)]]).T
    
def wrap(angle):
    while angle > np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle
def saturate(value, minimum, maximum):
    return min(maximum, max(value, minimum))