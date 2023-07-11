"""
    Visualization of UAV and moving obstacles
"""

import numpy as np
import matplotlib.pyplot as plt
from msg.threeDState import ThreeDState
from typing import List

class ThreeDViz:

    def __init__(self, limits) -> None: # limits of the form [xlimits, ylimits] where xlimits and ylimits are of the form [min,max]
        """
            Sets up the visualization
        """
        plt.ion()
        self.limits=limits
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(projection="3d")


    def update(self, uav_state:ThreeDState, target_states:List[ThreeDState]):
        """
            Updates the visualization with the new UAV and target positions.
        """
        self._ax.clear()
        uav_pos = np.reshape(uav_state.getPos(),(3))
        uav_vel = np.reshape(uav_state.toArray()[3:],(3))
        self._ax.scatter(uav_pos[0],uav_pos[1], zs=uav_pos[2], zdir='z',c='r')
        self._ax.quiver([uav_pos[0]], [uav_pos[1]], [uav_pos[2]], [uav_pos[0]+uav_vel[0]], [uav_pos[1]+uav_vel[1]], [uav_pos[2]+uav_vel[2]], colors='r')
        x = []
        y = []
        z = []
        for state in target_states:
            pos = np.reshape(state.getPos(),(3))
            vel = np.reshape(state.toArray()[3:], (3))
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])
            self._ax.quiver([pos[0]], [pos[1]], [pos[2]], [pos[0]+vel[0]], [pos[1]+vel[1]], [pos[2]+vel[2]],colors='b')
        self._ax.scatter(x,y, zs=z, zdir='z',c='b')
        self._ax.set_title("UAV and Targets")
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")
        self._ax.set_zlabel("z")
        self._ax.set_xlim(self.limits[0])
        self._ax.set_ylim(self.limits[1])
        
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()