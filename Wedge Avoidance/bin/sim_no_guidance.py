import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1])) 
import matplotlib.pyplot as plt
from viewers.world_viewer import WorldViewer
import parameters.simulation_parameters as SIM
from models.mav_dynamics import MavDynamics
from guidance.no_guidance import Guidance

# initialize view  er
world_view = WorldViewer()
# initialize vehicles
ownship = MavDynamics(SIM.ownship0)
intruders = []
for intruder0 in SIM.intruders0: 
    intruders.append(MavDynamics(intruder0))
# initialize guidance algorithm
guidance = Guidance()
bearings, sizes = ownship.camera(intruders)
# initialize the simulation time
sim_time = SIM.start_time
# main simulation loop
while sim_time < SIM.end_time:
    # Propagate dynamics in between plot samples
    t_next_plot = sim_time + SIM.ts_plotting
    while sim_time < t_next_plot:  # updates control and dynamics at faster simulation rate
        # get camera measurements
        bearings, sizes = ownship.camera(intruders)
        # compute guidance action
        u = guidance.update(ownship.state, bearings, sizes)
        # move ownship
        ownship.update(u)
        # move intruders
        for intruder in intruders:
            intruder.update(0.)
        sim_time += SIM.ts_simulation
    # -------update viewers-------------
    world_view.update(ownship, intruders, bearings, sizes)
    plt.pause(0.0001)  # the pause causes the figure to be displayed during the simulation

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()



