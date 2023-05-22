"""
    Visualization of UAV and moving obstacles
"""

import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence

class twoDViz:

    def __init__(self) -> None:
        """
            Sets up the visualization
        """
        self._fig, self._ax = plt.subplots()
        self._ax.set_title("UAV and Targets")
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")



    def update(self, uav_pos:np.ndarray, target_pos:Sequence[np.ndarray]):
        """
            Updates the visualization with the new UAV and target positions.
        """
        self._ax.clear()
        self._ax.scatter(uav_pos[0],uav_pos[1],c='r')
        x = []
        y = []
        for pos in target_pos:
            x.append(pos[0])
            y.append(pos[1])
        self._ax.scatter(x,y,c='b')