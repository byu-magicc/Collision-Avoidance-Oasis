"""
    Class to simulate a camera by giving a bearing to the targets
"""

import numpy as np
from typing import List

class UnitVectorSensor:
    def __init__(self) -> None:
        pass

    def update(self, uav_position:np.ndarray, target_positions:List[np.ndarray])->List[np.ndarray]:
        unit_vectors = []
        for target_pos in target_positions:
            dif = target_pos - uav_position
            dif /= np.linalg.norm(dif)
            unit_vectors.append(dif)
        return unit_vectors