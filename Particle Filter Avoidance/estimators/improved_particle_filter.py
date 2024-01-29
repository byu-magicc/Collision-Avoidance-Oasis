

import numpy as np
from scipy.stats import norm
from numpy import sin, cos
from copy import deepcopy
from msg.twoDYawState import TwoDYawState
from msg.bearing_msg import BearingMsg

class TTCParticleFilter:
    def __init__(self, initial_bearing, initial_yaw, ts) -> None:
        self.ts = ts
        self.num_particles = 1000
        self.bearing_std = 0.001
        self.Rinv = np.diag([1/self.bearing_std**2, 1/self.bearing_std**2])
        self.tau_pr_noise = 0.5
        self.vi_pr_noise = 5.
        self.yaw_pr_noise = 1.

        self.tau_res_std = 0.1*self.ts
        self.vi_res_std = 0.5*self.ts
        self.yaw_res_std = 0.05*self.ts

        self.L = np.array([[0.,0.],
                           [0.,0.],
                           [1.,0.],
                           [0.,1.],
                           [0.,0.]]) * np.sqrt(self.ts)
        tau_min = 5
        tau_max = 100
        vi_max = 50
        vi_min = 2
        self.xhats = np.zeros((5, self.num_particles))
        self.weights = np.zeros((self.num_particles))
        for i in range(self.num_particles):
            self.xhats[:,i] = np.array([initial_bearing,#np.random.normal(initial_bearing, self.bearing_std),
                                (tau_max - tau_min)*np.random.rand()+tau_min,
                                (vi_max-vi_min)*np.random.rand()+vi_min,
                                2*np.pi*np.random.rand(),
                                initial_yaw,#np.random.normal(initial_yaw, self.yaw_std)
                                ])

    def update(self, measurement:BearingMsg, state:TwoDYawState, input:float):
        self.propagate_model(state, input)
        # self.measurement_update(measurement)
        # self.resample(measurement)

    def propagate_model(self, state:TwoDYawState, input:float):
        for i in range(self.num_particles):
            self.xhats[:,i] = self.update_particle(self.xhats[:,i], state, input)

    def update_particle(self, xhat, state:TwoDYawState):
        x1 = self._f(xhat, state, input)
        x2 = self._f(xhat + self.ts/2.*x1, state, input)
        x3 = self._f(xhat + self.ts/2*x2, state, input)
        x4 = self._f(xhat + self.ts*x3, state, input)

        xhat += np.reshape(self.ts/6.*(x1+2*x2+2*x3+x4) + self.L @ np.array([[np.sqrt(self.vi_pr_noise)*np.random.rand(), np.sqrt(self.yaw_pr_noise)*np.random.rand()]]).T, xhat.shape) 
        return xhat
    
    def measurement_update(self, measurement:BearingMsg):
        for i in range(self.num_particles):
            y = np.array([[measurement.bearing, measurement.yaw]]).T
            h = np.array([[self.xhats[:,i].item(0), self.xhats[:,i].item(4)]]).T
            self.weights[i] = np.exp(-1/2. * (y-h).T @ self.Rinv @ (y-h)).item(0)
        # normalize the weights
        self.weights /= np.sum(self.weights)


    def resample(self, measurement:BearingMsg):
        old_particles = deepcopy(self.xhats)
        old_weights = deepcopy(self.weights)

        rr = np.random.rand()/self.num_particles
        i = 0 # particle to be sampled
        cc = old_weights[i]
        for mm in range(self.num_particles):
            u = rr + (mm-1)/self.num_particles
            while u > cc:
                i += 1
                cc += old_weights[i]
            self.xhats[:,mm] = old_particles[:,i]# + np.array([0., np.random.normal(0, self.tau_res_std), np.random.normal(0,self.vi_res_std), np.random.normal(0,self.yaw_res_std), 0.])
            self.xhats[0,mm] = measurement.bearing
            self.xhats[4,mm] = measurement.yaw
            self.weights[mm] = 1#old_weights[i]

        # renormalize after sampling
        #self.weights /= np.sum(self.weights)

    def _f(self, x, state, input):
        # get values needed for the calculation
        vi = x[2:]
        # calculate xdot
        xdot = np.zeros((4,1))
        xdot[0:2] = vi
        return xdot

    def get_particle_states(self, uav_state:TwoDYawState):
        states = []
        for i in range(self.num_particles):
            states.append(self.get_particle_state(self.xhats[:,i], self.weights[i], uav_state))
        return states
    
    def get_particle_state(self, xhat, weight, uav_state:TwoDYawState):
        theta = xhat.item(4)+xhat.item(0)
        vo = uav_state.vel
        d = xhat.item(1)*vo
        xpos = d*np.sin(theta) + uav_state.getPos().item(0)
        ypos = d*np.cos(theta) + uav_state.getPos().item(1)
        state = TwoDYawState(xpos, ypos, xhat.item(3), xhat.item(2))
        state.weight = weight
        return state

def wrap(diff):
    while diff > np.pi:
        diff -= 2*np.pi
    while diff < -np.pi:
        diff += 2*np.pi
    return diff