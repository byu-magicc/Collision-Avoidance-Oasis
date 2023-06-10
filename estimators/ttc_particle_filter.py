

import numpy as np
from numpy import sin, cos
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

class TTCParticleFilter:
    def __init__(self) -> None:
        self.num_particles = 50
        self.particles = []

    def update(self, measurement:BearingMsg, state:TwoDYawState, input):
        pass

    def propagate_model(self, state:TwoDYawState, input):
        for particle in self.particles:
            particle.update(state, input)

