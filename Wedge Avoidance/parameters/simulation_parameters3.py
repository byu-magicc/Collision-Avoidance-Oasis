import numpy as np
from message_types.msg_state import MsgState

ts_simulation = 0.01  # time step for simulation
start_time = 0.  # start time for simulation
end_time = 20.  # end time for simulation
ts_plotting = 0.1  # refresh rate for plots

# drawing parameters
world_size = 1000
uav_size = 2

# set initial conditions for ownship and intruders
ownship0 = MsgState(pos=np.array([[0.], [0.]]), vel=20., theta=0.)
intruders0 = []
intruders0.append(MsgState(pos=np.array(
    [[100.], [100.]]), vel=30., theta=-np.pi / 2))
