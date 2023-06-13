"""
    Visualization of UAV and moving obstacles
"""

import numpy as np
import matplotlib.pyplot as plt
from msg.twoDYawState import TwoDYawState
from typing import List

class twoDVizWithParticles:

    def __init__(self, limits) -> None: # limits of the form [xlimits, ylimits] where xlimits and ylimits are of the form [min,max]
        """
            Sets up the visualization
        """
        plt.ion()
        self.limits=limits
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)


    def update(self, uav_state:TwoDYawState, target_state:TwoDYawState, particle_states:List[TwoDYawState]):
        """
            Updates the visualization with the new UAV and target positions.
        """
        self._ax.clear()
        self._ax.set_xlim(self.limits[0])
        self._ax.set_ylim(self.limits[1])

        uav_pos = np.reshape(uav_state.getPos(),(2))
        x = []
        y = []
        sizes = []
        alpha = 2
        for particle in particle_states:
            pos = np.reshape(particle.getPos(),(2))
            x.append(pos[0])
            y.append(pos[1])
            sizes.append(particle.weight*alpha)

        # plot the particle positions
        self._ax.scatter(x,y,c='b',s=sizes)

        # plot the target's actual position and velocity arrow
        pos = np.reshape(target_state.getPos(),(2))
        self._ax.arrow(pos[0], pos[1], target_state.vel*np.sin(target_state.yaw),target_state.vel*np.cos(target_state.yaw),color='r')
        self._ax.scatter(pos[0],pos[1],c='r')

        #plot the UAV's position and velocity arrow
        self._ax.scatter(uav_pos[0],uav_pos[1],c='g')
        self._ax.arrow(uav_pos[0], uav_pos[1], uav_state.vel*np.sin(uav_state.yaw), uav_state.vel*np.cos(uav_state.yaw), color='g')

        # make the plot look pretty
        self._ax.set_title("UAV and Targets")
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")
        
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()