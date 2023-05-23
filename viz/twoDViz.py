"""
    Visualization of UAV and moving obstacles
"""

import numpy as np
import matplotlib.pyplot as plt
from msg.twoDYawState import TwoDYawState
from typing import List

class twoDViz:

    def __init__(self, limits) -> None: # limits of the form [xlimits, ylimits] where xlimits and ylimits are of the form [min,max]
        """
            Sets up the visualization
        """
        plt.ion()
        self.limits=limits
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)


    def update(self, uav_state:TwoDYawState, target_states:List[TwoDYawState]):
        """
            Updates the visualization with the new UAV and target positions.
        """
        self._ax.clear()
        uav_pos = np.reshape(uav_state.getPos(),(2))
        self._ax.scatter(uav_pos[0],uav_pos[1],c='r')
        self._ax.arrow(uav_pos[0], uav_pos[1], uav_state.vel*np.sin(uav_state.yaw), uav_state.vel*np.cos(uav_state.yaw), color='r')
        x = []
        y = []
        for state in target_states:
            pos = np.reshape(state.getPos(),(2))
            x.append(pos[0])
            y.append(pos[1])
            self._ax.arrow(pos[0], pos[1], state.vel*np.sin(state.yaw),state.vel*np.cos(state.yaw),color='b')
        self._ax.scatter(x,y,c='b')
        self._ax.set_title("UAV and Targets")
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")
        self._ax.set_xlim(self.limits[0])
        self._ax.set_ylim(self.limits[1])
        
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()