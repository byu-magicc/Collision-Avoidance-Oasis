

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
        self.bearing_std = 0.01
        self.yaw_std = 0.01
<<<<<<< HEAD
        self.tau_resample_std = 0.5
        self.vi_resample_std = 0.5
        self.yaw_resample_std = 0.1
        self.Rinv = np.diag([1/self.bearing_std**2, 1/self.yaw_std**2])
        self.Qinv = np.diag([1/self.bearing_std**2, 1/self.tau_resample_std**2, 1/self.vi_resample_std**2, 1/self.yaw_resample_std**2, 1/self.tau_resample_std**2])
=======
        self.Rinv = np.diag([1/self.bearing_std**2, 1/self.yaw_std**2])
        self.tau_pr_noise = 0.1
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
>>>>>>> d6923400762a057f1575cd1dc3d2b70b0a63806b
        tau_min = 5
        tau_max = 100
        vi_max = 50
        vi_min = 2
        self.xhats = np.zeros((5, self.num_particles))
        self.weights = np.zeros((self.num_particles))
        for i in range(self.num_particles):
<<<<<<< HEAD
            particle = Particle(np.random.normal(initial_bearing, self.bearing_std),#initial_bearing,#
                                (tau_max - tau_min)*np.random.rand()+tau_min,
                                (vi_max-vi_min)*np.random.rand()+vi_min,
                                2*np.pi*np.random.rand(),
                                np.random.normal(initial_yaw, self.yaw_std), #initial_yaw,#
                                ts)
            self.particles.append(particle)
=======
            self.xhats[:,i] = np.array([initial_bearing,#np.random.normal(initial_bearing, self.bearing_std),
                                (tau_max - tau_min)*np.random.rand()+tau_min,
                                (vi_max-vi_min)*np.random.rand()+vi_min,
                                2*np.pi*np.random.rand(),
                                initial_yaw,#np.random.normal(initial_yaw, self.yaw_std)
                                ])
>>>>>>> d6923400762a057f1575cd1dc3d2b70b0a63806b

    def update(self, measurement:BearingMsg, state:TwoDYawState, input:float):
        self.propagate_model(state, input)
        # self.measurement_update(measurement)
        # self.resample(measurement)

    def propagate_model(self, state:TwoDYawState, input:float):
        for i in range(self.num_particles):
            self.xhats[:,i] = self.update_particle(self.xhats[:,i], state, input)

    def update_particle(self, xhat, state:TwoDYawState, input):
        x1 = self._f(xhat, state, input)
        x2 = self._f(xhat + self.ts/2.*x1, state, input)
        x3 = self._f(xhat + self.ts/2*x2, state, input)
        x4 = self._f(xhat + self.ts*x3, state, input)

        xhat += np.reshape(self.ts/6.*(x1+2*x2+2*x3+x4) + self.L @ np.array([[np.sqrt(self.vi_pr_noise)*np.random.rand(), np.sqrt(self.yaw_pr_noise)*np.random.rand()]]).T, xhat.shape) 
        return xhat
    
    def measurement_update(self, measurement:BearingMsg):
        for i in range(self.num_particles):
            y = np.array([[measurement.bearing, measurement.yaw]]).T
<<<<<<< HEAD
            h = np.array([[particle.xhat.item(0), particle.xhat.item(4)]]).T
            particle.weight *= np.exp(-1/2. * (y-h).T @ self.Rinv @ (y-h)).item(0)


    def resample(self, measurement:BearingMsg):
        sum_of_weights = 0
        weights = []
        for particle in self.particles:
            sum_of_weights += particle.weight
            weights.append(particle.weight)
        
        weights = np.array(weights)
        norm_weights = weights/sum_of_weights
        norm_weights[-1] = 1.
        for i in range(1,self.num_particles):
            norm_weights[i] += norm_weights[i-1]
        rand = np.linspace(0, 1, self.num_particles)#np.random.rand(self.num_particles)
        # rand = np.sort(rand)

        old_particles = deepcopy(self.particles)
        self.particles.clear()

        i = 0 # random number
        j = 0 # particle to be sampled
        while i < self.num_particles:
            while rand[i] > norm_weights[j]:
                j += 1
            old_particle = old_particles[j]
            particle = Particle(np.random.normal(old_particle.xhat.item(0), self.bearing_std),#measurement.bearing,#
                                np.random.normal(old_particle.xhat.item(1), self.tau_resample_std),
                                np.random.normal(old_particle.xhat.item(2), self.vi_resample_std),
                                np.random.normal(old_particle.xhat.item(3), self.yaw_resample_std),
                                np.random.normal(old_particle.xhat.item(4), self.yaw_std),#measurement.yaw,#
                                self.ts)
            # particle.weight = np.exp(-1/2*(particle.xhat - old_particle.xhat).T @ self.Qinv @ (particle.xhat - old_particle.xhat)).item(0)
            self.particles.append(particle)
            i += 1
=======
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
>>>>>>> d6923400762a057f1575cd1dc3d2b70b0a63806b

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