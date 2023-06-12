

import numpy as np
from scipy.stats import chi2
from numpy import sin, cos
from copy import deepcopy
from msg.twoDYawState import TwoDYawState
from msg.bearing_msg import BearingMsg

class Particle:
    def __init__(self, eta, tau, vi, yawi, yawo, ts) -> None:
        self.xhat = np.array([[eta, tau, vi, yawi, yawo]]).T
        self.ts = ts
        self.weight = 1

    def update(self, state:TwoDYawState, input):
        x1 = self._f(self.xhat, state, input)
        x2 = self._f(self.xhat + self.ts/2.*x1, state, input)
        x3 = self._f(self.xhat + self.ts/2*x2, state, input)
        x4 = self._f(self.xhat + self.ts*x3, state, input)

        self.xhat += self.ts/6.*(x1+2*x2+2*x3+x4)


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
    
    def get_state(self, uav_state:TwoDYawState):
        theta = self.xhat.item(4)+self.xhat.item(0)
        vo = uav_state.vel
        d = self.xhat.item(1)*vo
        xpos = d*np.sin(theta) + uav_state.getPos().item(0)
        ypos = d*np.cos(theta) + uav_state.getPos().item(1)
        return TwoDYawState(xpos, ypos, self.xhat.item(3),self.xhat.item(2))

class TTCParticleFilter:
    def __init__(self, initial_bearing, initial_yaw, ts) -> None:
        self.ts = ts
        self.num_particles = 500
        self.particles = []
        self.bearing_std = 0.0001
        self.yaw_std = 0.01
        self.tau_resample_std = 1
        self.vi_resample_std = 1
        self.yaw_resample_std = 0.1
        tau_min = 5
        tau_max = 100
        vi_max = 130
        vi_min = 2
        for i in range(self.num_particles):
            particle = Particle(initial_bearing,#np.random.normal(initial_bearing, self.bearing_std),
                                (tau_max - tau_min)*np.random.rand()+tau_min,
                                (vi_max-vi_min)*np.random.rand()+vi_min,
                                2*np.pi*np.random.rand(),
                                initial_yaw,#np.random.normal(initial_yaw, self.yaw_std), 
                                ts)
            self.particles.append(particle)

    def update(self, measurement:BearingMsg, state:TwoDYawState, input:float):
        self.propagate_model(state, input)
        self.measurement_update(measurement, state, input)
        self.resample(measurement)

    def propagate_model(self, state:TwoDYawState, input:float):
        for particle in self.particles:
            particle.update(state, input)
    
    def measurement_update(self, measurement:BearingMsg, state:TwoDYawState, input:float):
        for particle in self.particles:
            particle.weight = chi2.pdf(np.linalg.norm(np.array([wrap(measurement.bearing-particle.xhat.item(0)), wrap(measurement.yaw-particle.xhat.item(4))])), df=2)

    def resample(self, measurement:BearingMsg):
        sum_of_weights = 0
        weights = []
        for particle in self.particles:
            sum_of_weights += particle.weight
            weights.append(particle.weight)
        
        weights = np.array(weights)
        norm_weights = weights/sum_of_weights
        for i in range(1,self.num_particles):
            norm_weights[i] += norm_weights[i-1]
        rand = np.random.rand(self.num_particles)
        rand = np.sort(rand)

        old_particles = deepcopy(self.particles)
        self.particles.clear()

        i = 0 # random number
        j = 0 # particle to be sampled
        while i < self.num_particles:
            while rand[i] > norm_weights[j]:
                j += 1
            old_particle = old_particles[j]
            particle = Particle(measurement.bearing,#np.random.normal(old_particle.xhat.item(0), self.bearing_std),
                                np.random.normal(old_particle.xhat.item(1), self.tau_resample_std),
                                np.random.normal(old_particle.xhat.item(2), self.vi_resample_std),
                                np.random.normal(old_particle.xhat.item(3), self.yaw_resample_std),
                                measurement.yaw,#np.random.normal(old_particle.xhat.item(4), self.yaw_std),
                                self.ts)
            self.particles.append(particle)
            i += 1

    def get_particle_states(self, uav_state:TwoDYawState):
        states = []
        for particle in self.particles:
            states.append(particle.get_state(uav_state))
        return states

def wrap(diff):
    while diff > np.pi:
        diff -= 2*np.pi
    while diff < -np.pi:
        diff += 2*np.pi
    return diff