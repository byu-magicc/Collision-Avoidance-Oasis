import numpy as np
import matplotlib.pyplot as plt
from msg.threeDState import ThreeDState
from msg.bearing_msg import BearingMsg
from typing import List

class PLKFViz:

    def __init__(self, ac_vel) -> None:
        
        plt.ion()
        self.fig = plt.figure()
        self.x_ax = self.fig.add_subplot(321)
        self.y_ax = self.fig.add_subplot(323)
        self.z_ax = self.fig.add_subplot(325)
        self.vx_ax = self.fig.add_subplot(322)
        self.vy_ax = self.fig.add_subplot(324)
        self.vz_ax = self.fig.add_subplot(326)
        self._t = []
        self.es_x = []
        self.es_y = []
        self.es_z = []
        self.es_vx = []
        self.es_vy = []
        self.es_vz = []
        self.ac_x = []
        self.ac_y = []
        self.ac_z = []
        self.ac_vx = ac_vel.item(0)
        self.ac_vy = ac_vel.item(1)
        self.ac_vz = ac_vel.item(2)

    def update(self, uav_state:ThreeDState, target_state:ThreeDState, xhat, t):
        target_xhat = xhat + uav_state.toArray()
        self._t.append(t)
        self.es_x.append(target_xhat[0,0])
        self.ac_x.append(target_state.xpos)
        self.es_y.append(target_xhat[1,0])
        self.ac_y.append(target_state.ypos)
        self.es_z.append(target_xhat[2,0])
        self.ac_z.append(target_state.zpos)
        self.es_vx.append(target_xhat[3,0])
        self.es_vy.append(target_xhat[4,0])
        self.es_vz.append(target_xhat[5,0])

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

        self.z_ax.clear()
        self.z_ax.plot(self._t, self.es_z, c='b')
        self.z_ax.plot(self._t, self.ac_z, c='r')
        self.z_ax.set_ylabel("Z Position")
        self.z_ax.set_xlabel("t")

        self.vx_ax.clear()
        self.vx_ax.plot(self._t, self.es_vx,c='b', label="Estimated")
        self.vx_ax.plot(self._t, [self.ac_vx]*len(self._t),c='r', label="Actual")
        self.vx_ax.set_ylabel("X Velocity")
        self.vx_ax.legend()

        self.vy_ax.clear()
        self.vy_ax.plot(self._t, self.es_vy,c='b')
        self.vy_ax.plot(self._t, [self.ac_vy]*len(self._t),c='r')
        self.vy_ax.set_ylabel("Y Velocity")

        self.vz_ax.clear()
        self.vz_ax.plot(self._t, self.es_vz,c='b')
        self.vz_ax.plot(self._t, [self.ac_vz]*len(self._t),c='r')
        self.vz_ax.set_ylabel("Z Velocity")
        self.vz_ax.set_xlabel("t")

        
