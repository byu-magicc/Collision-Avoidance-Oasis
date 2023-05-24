import numpy as np
import matplotlib.pyplot as plt
from msg.twoDYawState import TwoDYawState
from msg.bearing_msg import BearingMsg
from typing import List

class TwoDPosEsViz:

    def __init__(self, ac_vel, ac_psii) -> None:
        
        plt.ion()
        self.fig = plt.figure()
        self.x_ax = self.fig.add_subplot(411)
        self.y_ax = self.fig.add_subplot(414)
        self.vel_ax = self.fig.add_subplot(413)
        self.psii_ax = self.fig.add_subplot(414)
        self._t = []
        self.es_x = []
        self.es_y = []
        self.es_vel = []
        self.es_psii = []
        self.ac_x = []
        self.ac_y = []
        self.ac_vel = ac_vel
        self.ac_psii = ac_psii

    def update(self, uav_state:TwoDYawState, target_state:TwoDYawState, target_xhat, t, bearing_msg:BearingMsg):

        self._t.append(t)
        self.es_x.append(target_xhat[0,0])
        self.ac_x.append(target_state.xpos)
        self.es_y.append(target_xhat[1,0])
        self.ac_y.append(target_state.ypos)
        self.es_vel.append(target_xhat[2,0])
        self.es_psii.append(target_xhat[3,0])

    def update_plots(self):
        self.x_ax.clear()
        self.x_ax.set_title("Position Estimator")
        self.x_ax.plot(self._t, self.es_x, c='b')
        self.x_ax.plot(self._t, self.ac_x, c='r')
        self.x_ax.set_ylabel("X Position")

        self.y_ax.clear()
        self.y_ax.plot(self._t, self.es_y, c='b')
        self.y_ax.plot(self._t, self.ac_y, c='r')
        self.y_ax.set_ylabel("Y Position")

        self.vel_ax.clear()
        self.vel_ax.plot(self._t, self.es_vel, c='b')
        self.vel_ax.plot(self._t, [self.ac_vel]*len(self._t), c='r')
        self.vel_ax.set_ylabel("Velocity")

        self.psii_ax.clear()
        self.psii_ax.plot(self._t, self.es_psii,c='b')
        self.psii_ax.plot(self._t, [self.ac_psii]*len(self._t),c='r')
        self.psii_ax.set_ylabel("Target Yaw")
        self.psii_ax.set_xlabel("t")

        
