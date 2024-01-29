import numpy as np
import matplotlib.pyplot as plt
from msg.twoDYawState import TwoDYawState
from msg.bearing_msg import BearingMsg
from typing import List

class TwoDEstimatorViz:

    def __init__(self, ac_vel, ac_psii) -> None:
        
        plt.ion()
        self.fig = plt.figure()
        self._tau_ax = self.fig.add_subplot(411)
        self._vel_ax = self.fig.add_subplot(412)
        self._psii_ax = self.fig.add_subplot(413)
        self._eta_ax = self.fig.add_subplot(414)
        self._t = []
        self.es_tau = []
        self.es_vel = []
        self.es_psii = []
        self.es_eta = []
        self.ac_tau = []
        self.ac_eta = []
        self.ac_vel = ac_vel
        self.ac_psii = ac_psii

    def update(self, uav_state:TwoDYawState, target_state:TwoDYawState, target_xhat, t, bearing_msg:BearingMsg):
        
        dif = np.reshape(target_state.getPos() - uav_state.getPos(),(2))
        tau = np.linalg.norm(dif)/uav_state.vel

        self.es_tau.append(target_xhat[1,0])
        self.ac_tau.append(tau)
        self._t.append(t)
        self.es_vel.append(target_xhat[2,0])
        self.es_psii.append(target_xhat[3,0])
        self.es_eta.append(target_xhat[0,0])
        self.ac_eta.append(bearing_msg.bearing)

    def update_plots(self):
        self._tau_ax.clear()
        self._tau_ax.plot(self._t, self.es_tau, label="Estimated", c='b')
        self._tau_ax.plot(self._t, self.ac_tau, label="Actual", c='r')
        self._tau_ax.legend()
        self._tau_ax.set_ylabel("Tau")
        self._tau_ax.set_title("TTC EKF")

        self._vel_ax.clear()
        self._vel_ax.plot(self._t, self.es_vel, c='b')
        self._vel_ax.plot(self._t, [self.ac_vel]*len(self._t), c='r')
        self._vel_ax.set_ylabel("Velocity")

        self._psii_ax.clear()
        self._psii_ax.plot(self._t, self.es_psii,c='b')
        self._psii_ax.plot(self._t, [self.ac_psii]*len(self._t),c='r')
        self._psii_ax.set_ylabel("Target Yaw")

        self._eta_ax.clear()
        self._eta_ax.plot(self._t, self.es_eta, c='b')
        self._eta_ax.plot(self._t, self.ac_eta, c='r')
        self._eta_ax.set_ylabel("Bearing")
        self._eta_ax.set_xlabel("t")

        
