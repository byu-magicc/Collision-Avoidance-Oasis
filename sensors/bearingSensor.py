"""
    Class to simulate a camera by giving a bearing to the targets
"""

import numpy as np
from typing import List
from msg.bearing_msg import BearingMsg

class BearingSensor:
    def __init__(self) -> None:
        pass

    def update(self, uav_position:np.ndarray, uav_yaw:float, target_positions:List[np.ndarray])->List[BearingMsg]:
        bearings = []
        for target_pos in target_positions:
            dif = target_pos - uav_position
            bearing = np.arctan2(dif[0], dif[1])
            bearings.append(BearingMsg(bearing.item(0), uav_yaw))
        return bearings