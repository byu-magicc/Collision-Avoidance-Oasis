import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

# define constraints for the optimizer to use later on
# minimum and maximum ranges of detection
r_min = 10
r_max = 1000

# maximum predicted velocity
v_max = 90


po=np.array([[0.,0.]]).T
vo=np.array([[0.,20.]]).T

# generate an example trajectory to calculate LOS vectors and TTC
pi=np.array([[-0.,100.]]).T
vi=np.array([[-5.,-20.]]).T

# generate LOS vectors from example
l1=pi-po
l1/=np.linalg.norm(l1)

ts=0.1
l2 = pi+vi*ts - po-vo*ts
l2 /= np.linalg.norm(l2)

# calculate the TTC (assume the camera is facing line of travel)
ec=vo/np.linalg.norm(vo)
tau = ((po-pi).T @ ec)/((vi-vo).T @ ec)

# get the unit vector perpendicular to the camera normal vector
ecp = np.array([[0, -1],[1,0]]) @ ec

# solve the min-norm problem
A1 = np.concatenate([l1,-l2,np.zeros((2,1)),np.eye(2)*ts],axis=1)
A2 = np.concatenate([l1,np.zeros((2,1)),-ecp,np.eye(2)*tau], axis=1)
A = np.concatenate([A1,A2],axis=0)

b = np.concatenate([vo*ts,vo*tau],axis=0)

x = np.linalg.pinv(A)@b

# set the result up as a possible trajectory to plot along 
# with the original example
vi2 = x[3:]
a1 = x.item(0)
pi2 = l1*a1+po

# run an optimizer to get the closest and furthest starting 
# points possible while respecting the constraints above

# function to minimize alpha
def f_min(x):
    return x
# function to maximize alpha
def f_max(x):
    return -x
# function for maximum velocity constraint
def max_vel(x):
    pi2t = x*((pi2+vi2*tau)-(po+vo*tau))+(po+vo*tau)
    vel = (pi2t-pi2)/tau
    return v_max - np.linalg.norm(vel)
# function for maximum range constraint
r0 = np.linalg.norm(pi2-po)
def max_range(x):
    return r_max - r0*x
# function for minimum range constraint
def min_range(x):
    return r0*x - r_min

cons = ({'type':'ineq', 'fun':max_vel}, 
        {'type':'ineq', 'fun':max_range},
        {'type':'ineq', 'fun':min_range})

x0=1.

# do the optimization to minimize alpha
result = minimize(f_min,x0, constraints=cons)
p_min = result.x.item(0)*(pi2-po)+po
pi2t = result.x.item(0)*((pi2+vi2*tau)-(po+vo*tau))+(po+vo*tau)
v_min = (pi2t-p_min)/tau

# do the optimization to maximize alpha
result = minimize(f_max, x0, constraints=cons)
p_max = result.x.item(0)*(pi2-po)+po
pi2t = result.x.item(0)*((pi2+vi2*tau)-(po+vo*tau))+(po+vo*tau)
v_max = (pi2t-p_max)/tau

class Trajectories:
    def __init__(self, initial_pos, velocities) -> None:
        self.poss = initial_pos
        self.vels = velocities

    def update(self):
        for i in range(len(self.poss)):
            self.poss[i] += self.vels[i]*ts
    
    def get_Opos(self):
        return self.poss[0]
    
    def get_Iposes(self):
        return self.poss[1:]

class Plotter:
    def __init__(self, num_intruders, limits) -> None:
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.num_intruders = num_intruders
        self.limits=limits
        self.pox = []
        self.poy = []
        self.pix = []
        self.piy = []
        for i in range(num_intruders):
            self.pix.append([])
            self.piy.append([])

    def update_plot(self, own_pos, i_poses):
        self.ax.clear()
        # add the points to the respective lists
        self.pox.append(own_pos.item(0))
        self.poy.append(own_pos.item(1))
        for i in range(self.num_intruders):
            self.pix[i].append(i_poses[i].item(0))
            self.piy[i].append(i_poses[i].item(1))

        # assume the last two are the min and max range solutions
        # draw the rectangle of possible positions by connecting the points
        for i in range(len(self.pix[0])):
            self.ax.plot([self.pix[-2][i],self.pix[-1][i]],[self.piy[-2][i],self.piy[-1][i]],c='purple')

        # plot each of the actual intruders
        for i in range(self.num_intruders-2):
            self.ax.plot(self.pix[i],self.piy[i],label=f"Intruder {i+1}")

        # plot the min and max-range intruders
        self.ax.plot(self.pix[-2],self.piy[-2],label="Min Intruder")
        self.ax.plot(self.pix[-1],self.piy[-1],label="Max Intruder")

        # plot the own-ship path
        self.ax.plot(self.pox,self.poy,label='Own',c='r')

        # for i in range(self.num_intruders):
        #     self.ax.plot([self.pox[-1],self.pix[i][-1]],[self.poy[-1],self.piy[i][-1]],label=f"Bearing {i+1}")

        self.ax.set_title("Positions of Own-ship and Intruders")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.legend()
        # self.ax.set_xlim(self.limits[0])
        # self.ax.set_ylim(self.limits[1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

t=0.
ts = 0.1
traj = Trajectories([po, pi, p_min, p_max], [vo, vi, v_min, v_max])
plotter = Plotter(len(traj.get_Iposes()), [[-130,70],[-5,130]])
while t < tau:
    plotter.update_plot(traj.get_Opos(),traj.get_Iposes())
    traj.update()
    t+=ts
    time.sleep(ts)

plt.ioff()
plotter.update_plot(traj.get_Opos(), traj.get_Iposes())
plt.show()