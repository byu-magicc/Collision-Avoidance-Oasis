import os
import sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parents[1]))
from guidance.simple_controller import Controller
from guidance.path_manager import PathManager
from planning.rrt_dubins_planner import RRTDubins
from estimation.estimation_dyn_wedge import Estimation
from models.mav_dynamics import MavDynamics
import parameters.params_dyn_wedge_plan as SIM
from viewers.world_viewer_planner import WorldViewer
from message_types.msg_waypoints import MsgWaypoints
import matplotlib.pyplot as plt


def on_key(event):
    global simulation_is_paused
    if SIM.step_plot:
        if event.key == 'p' and not simulation_is_paused:
            simulation_is_paused = True
            print('paused')
        elif event.key == 'p' and simulation_is_paused:
            simulation_is_paused = False
            print('resumed')

simulation_is_paused = False
# initialize goal point (north, east)
goal = SIM.goal_point
# initialize viewer
world_view = WorldViewer(
    goal=goal, sim_settings=SIM)
plt.show()
plt.connect('key_press_event', on_key)
# initialize vehicles
ownship = MavDynamics(SIM.ownship0)
intruders = []
for intruder0 in SIM.intruders0:
    intruders.append(MavDynamics(intruder0))
# initialize estimation algorithm
estimator = Estimation(goal=goal, time_delta=SIM.estimation_settings['time_delta'],
                       future_steps=SIM.estimation_settings['future_steps'])
estimation = {}

# initialize the path manager
path_manager = PathManager(sim_settings=SIM)

# initialize the planner
rrt_dubins_planner = RRTDubins(sim_settings=SIM)
# initialize the waypoint list - this will get replaced as soon as we plan
waypoints = MsgWaypoints()

# initialize the controller
controller = Controller(sim_settings=SIM)

# initialize the simulation time
sim_time = SIM.start_time

# for the path planner
path_planned = False

# main simulation loop
while sim_time < SIM.end_time:

    # Propagate dynamics in between plot samples
    t_next_plot = sim_time + SIM.ts_plotting

    if not simulation_is_paused:
        # this is the main dynamics loop
        while (sim_time < t_next_plot):  # updates control and dynamics at faster simulation rate

            # ----------------------------------------------------------------------------------
            # begin this simulation time step by gathering the measured qualities
            # get camera measurements
            bearings, sizes = ownship.camera(intruders)

            # ----------------------------------------------------------------------------------
            # Next, do all estimations, using the measurement data
            # Generate the intruder estimate
            estimation = estimator.update(ownship.state, bearings, sizes)

            # This is where we would typically do the ownship state estimation, but we are skipping that for simplicity
            #
            # We might need to put in a re-planning strategy here in case the intruder measurements
            # do not match up with the estimates

            # ----------------------------------------------------------------------------------
            # Next, construct and manage the current path - a dubins path
            if not path_planned and estimation['has_collision']:
                path_planned = True
                # TODO: need to check this. I am not sure if the path manager requesting an update is working correctly
                # and I need to put in a good way to update the path if the estimation changes.
                # without re-finding the path every time step... which is how it currently works
                waypoints = rrt_dubins_planner.find_path(current_ownship=ownship,
                                                        estimation=estimation,
                                                        world_viewer=world_view)

            # get the latest path, based on the waypoints that the planner came up with
            path = path_manager.update(
                waypoints=waypoints, state=ownship.state)

            # ----------------------------------------------------------------------------------
            # Next, get the commands for the dynamics from the path follower
            u = controller.update(path=path, state=ownship.state)

            # ----------------------------------------------------------------------------------
            # Lastly, move ownship
            ownship.update(u)
            # and move intruders
            for intruder in intruders:
                intruder.update(0.0)

            # update the sim time
            sim_time += SIM.ts_simulation

        # -------update viewers-------------
        world_view.update(ownship, intruders, bearings, sizes, waypoints, estimation)
        # the pause causes the figure to be displayed during the simulation
        plt.pause(0.0001)
    else:
        plt.pause(0.5)


# Keeps the program from closing until the user presses a button.
print("Press key to close")
plt.waitforbuttonpress()
plt.close()


