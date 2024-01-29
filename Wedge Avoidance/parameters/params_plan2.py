import numpy as np
from message_types.msg_state import MsgState

# simulation parameters
ts_simulation = 0.01  # time step for simulation
start_time = 0.  # start time for simulation
end_time = 25.  # end time for simulation
ts_plotting = 0.1  # refresh rate for plots
step_plot = True # this allows the pausing ability

# drawing parameters
world_size = 500
uav_size = 2

# set initial conditions for ownship 
ownship_velocity = 0
ownship0 = MsgState(pos=np.array([[-80.], [0.]]), vel=ownship_velocity, theta=0.)

# set capabilities for the ownship
turn_radius = 40

# this disables the controller - 
# if this is true, the autopilot will not modify the heading of the ownship
control_disabled = False


# settings for intruders
intruders0 = []
intruders0.append(MsgState(pos=np.array(
    [[100.], [-250.]]), vel=30., theta=np.pi / 2))


# set the goal point in (north, east) coordinates
goal_point = np.array([[400], [0]])
end_pose = np.concatenate((goal_point, np.array([[0]])))

# parameters for the estimator and viewing options
estimation_settings = {
    'show_bounding_lines': False,
    'show_exact_estimates': False,
    'show_wedges': True,
    'time_delta': 1,
    'future_steps': 40
}
