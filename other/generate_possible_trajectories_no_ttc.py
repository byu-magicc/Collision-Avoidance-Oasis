import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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
pi=np.array([[-30.,100.]]).T
vi=np.array([[20.,0.]]).T

# generate LOS vectors from example
l1=pi-po
l1/=np.linalg.norm(l1)

ts=0.1
l2 = pi+vi*ts - po-vo*ts
l2 /= np.linalg.norm(l2)

timesteps_forward = 10

l3 = pi+vi*timesteps_forward*ts - po-vo*timesteps_forward*ts
l3 /= np.linalg.norm(l3)

# calculate the TTC (assume the camera is facing line of travel)
# ec=vo/np.linalg.norm(vo)
# tau = ((po-pi).T @ ec)/((vi-vo).T @ ec)

def calculate_positions_and_velocities(l1, l2, l3):

    # solve the min-norm problem
    A1 = np.concatenate([l1,-l2,np.zeros((2,1)),np.eye(2)*ts],axis=1)
    A2 = np.concatenate([l1,np.zeros((2,1)),-l3,np.eye(2)*timesteps_forward*ts], axis=1)
    A = np.concatenate([A1,A2],axis=0)

    b = np.concatenate([vo*ts,vo*timesteps_forward*ts],axis=0)

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
        vel = x*(vi2-vo)+vo
        return v_max - np.linalg.norm(vel)
    # function for maximum range constraint
    r0 = np.linalg.norm((pi2-po))
    def max_range(x):
        return r_max - x*r0
    # function for minimum range constraint
    def min_range(x):
        return x*r0 - r_min

    cons = ({'type':'ineq', 'fun':max_vel}, 
            {'type':'ineq', 'fun':max_range},
            {'type':'ineq', 'fun':min_range})

    x0=1.

    # do the optimization to minimize alpha
    result = minimize(f_min,x0, constraints=cons)
    p_min = result.x.item(0)*(pi2-po)+po
    pi2t = result.x.item(0)*((pi2+vi2*timesteps_forward*ts)-(po+vo*timesteps_forward*ts))+(po+vo*timesteps_forward*ts)
    vt_min = (pi2t-p_min)/(timesteps_forward*ts)

    # do the optimization to maximize alpha
    result = minimize(f_max, x0, constraints=cons)
    p_max = result.x.item(0)*(pi2-po)+po
    pi2t = result.x.item(0)*((pi2+vi2*timesteps_forward*ts)-(po+vo*timesteps_forward*ts))+(po+vo*timesteps_forward*ts)
    vt_max = (pi2t-p_max)/(timesteps_forward*ts)
    return (p_min, vt_min, p_max,vt_max)

pairs = []
labels = []

res = calculate_positions_and_velocities(l1, l2, l3)
pairs.append(res)
labels.append("Avg")

# rerun the calculations for l1 being slightly off
l1_minus = l1 / l1.item(1)
l1_minus[0,:]-=0.01
l1_minus /= np.linalg.norm(l1_minus)
# print("Angle Difference (minus) (degrees):")
# print(np.arccos(l1.T @ l1_minus)*180/np.pi)
# print()
res1m = calculate_positions_and_velocities(l1_minus, l2, l3)
pairs.append(res1m)
labels.append("1 Minus")

l1_plus = l1/l1.item(1)
l1_plus[0,:] += 0.01
l1_plus /= np.linalg.norm(l1_plus)
# print("Angle Difference (plus) (degrees):")
# print(np.arccos(l1.T@l1_plus)*180/np.pi)
# print()
res1p = calculate_positions_and_velocities(l1_plus, l2, l3)
pairs.append(res1p)
labels.append("1 Plus")

# # rerun the calculations for l2 being slightly off
# res2m = calculate_positions_and_velocities(l1, l2-np.array([[0.01, 0.01]]).T, ec, tau)
# pairs.append(res2m)
# labels.append("2 Minus")

# res2p = calculate_positions_and_velocities(l1, l2+np.array([[0.01, 0.01]]).T, ec, tau)
# pairs.append(res2p)
# labels.append("2 Plus")

# res_taum = calculate_positions_and_velocities(l1, l2, ec, tau-0.1)
# pairs.append(res_taum)
# labels.append("Tau Minus")

# res_taup = calculate_positions_and_velocities(l1, l2, ec, tau+0.1)
# pairs.append(res_taup)
# labels.append("Tau Plus")

class Trajectories:
    def __init__(self, initial_pos, velocities, pairs=[]) -> None:
        self.poss = initial_pos
        self.vels = velocities
        for pair in pairs:
            self.poss.append(pair[0])
            self.vels.append(pair[1])
            self.poss.append(pair[2])
            self.vels.append(pair[3])

    def update(self):
        for i in range(len(self.poss)):
            self.poss[i] += self.vels[i]*ts
    
    def get_Opos(self):
        return self.poss[0]
    
    def get_Iposes(self):
        return self.poss[1:]

class Plotter:
    def __init__(self, num_intruders, limits, num_pairs, pair_labels) -> None:
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.num_intruders = num_intruders
        self.limits=limits
        self.num_pairs = num_pairs
        self.pair_labels = pair_labels
        self.pox = []
        self.poy = []
        self.pix = []
        self.piy = []
        self.colors=["b", "orange", "g","yellow", "lime", "darkorange", "darkslategray", "olive", "orchid", "lawngreen", "aquamarine"]
        for i in range(num_intruders):
            self.pix.append([])
            self.piy.append([])

    def update_plot(self, own_pos, i_poses):
        head_width = 10
        self.ax.clear()
        # add the points to the respective lists
        self.pox.append(own_pos.item(0))
        self.poy.append(own_pos.item(1))
        for i in range(self.num_intruders):
            self.pix[i].append(i_poses[i].item(0))
            self.piy[i].append(i_poses[i].item(1))

        # draw the rectangle of possible positions by connecting the points
        for j in range(-self.num_pairs+1, 1):
            for i in range(len(self.pix[0])):
                self.ax.plot([self.pix[2*j-2][i],self.pix[2*j-1][i]],[self.piy[2*j-2][i],self.piy[2*j-1][i]],c='pink',zorder=-20)

        # plot each of the actual intruders
        color_i = 0
        for i in range(self.num_intruders-2*self.num_pairs):
            lines=self.ax.plot(self.pix[i],self.piy[i],label=f"Intruder {i+1}")
            if len(self.pix[i]) >= 2:
                self.ax.arrow(self.pix[i][0], self.piy[i][0], self.pix[i][1]-self.pix[i][0], self.piy[i][1]-self.piy[i][0], head_width=head_width, color=lines[0].get_color())
            color_i+=1

        # plot the min and max-range intruders
        for i in range(1-self.num_pairs, 1):
            lmin = self.ax.plot(self.pix[2*i-2],self.piy[2*i-2],label=self.pair_labels[i+self.num_pairs-1]+" Min")
            lmax = self.ax.plot(self.pix[2*i-1],self.piy[2*i-1],label=self.pair_labels[i+self.num_pairs-1]+" Max")
            if len(self.pix[2*i-2])>=2:
                self.ax.arrow(self.pix[2*i-2][0],self.piy[2*i-2][0], self.pix[2*i-2][1]-self.pix[2*i-2][0],self.piy[2*i-2][1]-self.piy[2*i-2][0], head_width=head_width, color=lmin[0].get_color())
                self.ax.arrow(self.pix[2*i-1][0],self.piy[2*i-1][0], self.pix[2*i-1][1]-self.pix[2*i-1][0],self.piy[2*i-1][1]-self.piy[2*i-1][0], head_width=head_width, color=lmax[0].get_color())
            color_i += 2
        # plot the own-ship path
        self.ax.plot(self.pox,self.poy,label='Own',c='r')
        if len(self.pox)>=2:
            self.ax.arrow(self.pox[0], self.poy[0], self.pox[1]-self.pox[0], self.poy[1]-self.poy[0], color='r', head_width=head_width)

        # for i in range(self.num_intruders):
        #     self.ax.plot([self.pox[-1],self.pix[i][-1]],[self.poy[-1],self.piy[i][-1]],label=f"Bearing {i+1}")

        self.ax.set_title("Positions of Own-ship and Intruders")
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.legend()
        # self.ax.set_xlim(self.limits[0])
        # self.ax.set_ylim(self.limits[1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def plot_interactive(self):
        TWOPI = 2*np.pi

        fig, ax = plt.subplots()

        t = np.arange(0.0, TWOPI, 0.001)
        initial_i = 0
        plots = []
        b=0
        for j in range(-self.num_pairs+1, 1):
            l, = plt.plot([self.pix[2*j-2][initial_i],self.pix[2*j-1][initial_i]],[self.piy[2*j-2][initial_i],self.piy[2*j-1][initial_i]], label=self.pair_labels[b],zorder=-20)
            plots.append(l)
            b+=1
        plt.legend()
        xlims = self.ax.get_xlim()
        ylims = self.ax.get_ylim()
        ax = plt.axis([xlims[0], xlims[1], ylims[0], ylims[1]])

        axamp = plt.axes([0.25, .03, 0.50, 0.02])
        # Slider
        samp = Slider(axamp, 'Amp', 0, len(self.pox)-1, valinit=0, valstep=1)

        def update(val):
            # amp is the current value of the slider
            i = samp.val
            # update curve
            b=0
            for j in range(-self.num_pairs+1, 1):
                plots[b].set_xdata([self.pix[2*j-2][i],self.pix[2*j-1][i]])
                plots[b].set_ydata([self.piy[2*j-2][i],self.piy[2*j-1][i]])
                b += 1
            # redraw canvas while idle
            fig.canvas.draw_idle()

        # call update function on slider value change
        samp.on_changed(update)

        plt.show()

t=0.
ts = 0.2
t_stop = 6.
traj = Trajectories([po, pi], [vo, vi], pairs)
plotter = Plotter(len(traj.get_Iposes()), [[-130,70],[-5,130]], len(pairs), labels)
while t < t_stop:
    plotter.update_plot(traj.get_Opos(),traj.get_Iposes())
    traj.update()
    t+=ts
    # time.sleep(ts)

plotter.update_plot(traj.get_Opos(), traj.get_Iposes())
plt.ioff()
plotter.plot_interactive()
plt.show()