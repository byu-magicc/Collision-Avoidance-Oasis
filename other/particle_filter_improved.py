import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time
from copy import deepcopy

# define constraints for the optimizer to use later on
# minimum and maximum ranges of detection
r_min = 10
r_max = 1000

# maximum predicted velocity
v_max = 90

desired_measurements = 2
if desired_measurements < 2:
    raise

po=np.array([[0.,0.]]).T
vo=np.array([[0.,20.]]).T

# generate an example trajectory to calculate LOS vectors and TTC
pi=np.array([[-30.,100.]]).T
vi=np.array([[20.,0.]]).T

# generate LOS vectors from example
lms = []

ts=0.2
for i in range(desired_measurements):
    lm = pi+vi*i*ts - po-vo*i*ts
    lm /= np.linalg.norm(lm)
    lms.append(lm)

# calculate the TTC (assume the camera is facing line of travel)
ec=vo/np.linalg.norm(vo)
tau = ((po-pi).T @ ec)/((vi-vo).T @ ec)

def calculate_trajectory_first(a1, ls, ec, tau):
    if len(ls)<2:
        raise
    l1 = ls[0]
    ls = ls[1:]
    # get the unit vector perpendicular to the camera normal vector
    ecp = np.array([[0, -1],[1,0]]) @ ec

    # solve the min-norm problem
    A=[]
    for i in range(len(ls)):
        Ap = []
        for j in range(i):
            Ap.append(np.zeros((2,1)))
        Ap.append(ls[i])
        for j in range(i+1,len(ls)+1):
            Ap.append(np.zeros((2,1)))
        Ap.append(-np.eye(2)*ts*(i+1))
        A1 = np.concatenate(Ap,axis=1)
        A.append(A1)
    A2 = []
    for i in range(len(ls)):
        A2.append(np.zeros((2,1)))
    A2.append(ecp)
    A2.append(-np.eye(2)*tau)
    A2 = np.concatenate(A2, axis=1)
    A.append(A2)
    A = np.concatenate(A,axis=0)

    b = []
    for i in range(len(ls)):
        b.append(-vo*(i+1)*ts+a1*l1)
    b.append(-vo*tau+a1*l1)
    b = np.concatenate(b,axis=0)

    x = np.linalg.pinv(A)@b

    # set the result up as a possible trajectory to plot along 
    # with the original example
    pi0 = a1*l1
    vi = x[-2:]
    return pi0, vi

def calculate_trajectory_last(at, ls, t_s, ec, tau):
    if len(ls)<2:
        raise
    lt = ls[-1]
    ls = ls[0:-1]
    t = t_s[-1]
    t_s = t_s[0:-1]
    # get the unit vector perpendicular to the camera normal vector
    ecp = np.array([[0, -1],[1,0]]) @ ec

    # solve the min-norm problem
    A=[]
    for i in range(len(ls)):
        Ap = []
        for j in range(i):
            Ap.append(np.zeros((2,1)))
        Ap.append(ls[i])
        for j in range(i+1,len(ls)+1):
            Ap.append(np.zeros((2,1)))
        Ap.append(np.eye(2)*(t-t_s[i]))
        A1 = np.concatenate(Ap,axis=1)
        A.append(A1)
    A2 = []
    for i in range(len(ls)):
        A2.append(np.zeros((2,1)))
    A2.append(ecp)
    A2.append(np.eye(2)*(t-tau))
    A2 = np.concatenate(A2, axis=1)
    A.append(A2)
    A = np.concatenate(A,axis=0)

    b = []
    for i in range(len(ls)):
        b.append(at*lt+vo*(t-t_s[i]))
    b.append(at*lt+vo*(t-tau))
    b = np.concatenate(b,axis=0)

    x = np.linalg.pinv(A)@b

    # set the result up as a possible trajectory to plot along 
    # with the original example
    pit = at*lt
    vi = x[-2:]
    return pit, vi


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
    def __init__(self, num_plot_i, num_intruders, limits) -> None:
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.num_plot_i = num_plot_i
        self.num_intruders = num_intruders
        self.limits=limits
        self.pox = []
        self.poy = []
        self.pix = []
        self.piy = []
        self.plot_x = []
        self.plot_y = []
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
            self.plot_x.append(i_poses[i].item(0))
            self.plot_y.append(i_poses[i].item(1))

        self.ax.scatter(self.plot_x, self.plot_y,s=1, color='g', zorder=-30)

        # plot each of the actual intruders
        color_i = 0
        for i in range(self.num_plot_i):
            lines=self.ax.plot(self.pix[i],self.piy[i],label=f"Intruder {i+1}")
            if len(self.pix[i]) >= 2:
                self.ax.arrow(self.pix[i][0], self.piy[i][0], self.pix[i][1]-self.pix[i][0], self.piy[i][1]-self.piy[i][0], head_width=head_width, color=lines[0].get_color())
            color_i+=1

        # plot the own-ship path
        self.ax.plot(self.pox,self.poy,label='Own',c='r')
        if len(self.pox)>=2:
            self.ax.arrow(self.pox[0], self.poy[0], self.pox[1]-self.pox[0], self.poy[1]-self.poy[0], color='r', head_width=head_width)

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

        initial_i = 0
        
        l,=plt.plot(self.plot_x[0:self.num_intruders], self.plot_y[0:self.num_intruders],marker='.', ls='', markersize=1)
        xlims = self.ax.get_xlim()
        ylims = self.ax.get_ylim()
        ax = plt.axis([xlims[0], xlims[1], ylims[0], ylims[1]])

        axamp = plt.axes([0.25, .03, 0.50, 0.02])
        # Slider
        samp = Slider(axamp, 'Timestep', 0, len(self.pox)-1, valinit=initial_i, valstep=1)

        def update(val):
            # amp is the current value of the slider
            i = samp.val
            # update curve
            l.set_xdata(self.plot_x[self.num_intruders*i:self.num_intruders*(i+1)])
            l.set_ydata(self.plot_y[self.num_intruders*i:self.num_intruders*(i+1)])
            # redraw canvas while idle
            fig.canvas.draw_idle()

        # call update function on slider value change
        samp.on_changed(update)

        plt.show()

num_particles = 5000
lms_norm = []
for lm in lms:
    lm_norm = lm / lm.item(1)
    lms_norm.append(lm_norm)
particle_p = deepcopy([po, pi])
particle_v = [vo, vi]
initialc = len(particle_p)
weights = []
pi0s = []
while len(particle_p)-initialc < num_particles:
    # randomly noisify the measurements
    lus = []
    for lm_norm in lms_norm:
        l_u = lm_norm
        l_u[0,0] += np.random.normal(0,0.001)
        l_u /= np.linalg.norm(l_u)
        lus.append(l_u)

    tau_u = tau + np.random.normal(0, 0.5)

    a1_u = (r_max-r_min)*np.random.random()+ r_min
    pos, vel = calculate_trajectory_first(a1_u, lus, ec, tau_u)
    if np.linalg.norm(vel)<=v_max:
        particle_p.append(pos)
        pi0s.append(pos)
        particle_v.append(vel)
        weights.append(1)

Rinv = np.diag([1/0.05**2, 1/.05**2])

def update_weights(weights, lm, particle_p, po):
    for i in range(len(particle_p)):
        phat = particle_p[i]-po
        phat /= np.linalg.norm(phat)
        weights[i] = np.exp(-1/2. * (lm-phat).T @ Rinv @ (lm-phat)).item(0)
    sum = np.sum(weights)
    weights = [x/sum for x in weights]
    return weights

def resample(pi0s, particle_p, particle_v, weights, t):
    old_pi0s = deepcopy(pi0s)
    old_weights = deepcopy(weights)

    rr = np.random.rand()/num_particles
    i = 0
    cc = old_weights[i]
    for mm in range(num_particles):
        u = rr + (mm-1)/num_particles
        while u > cc:
            i += 1
            cc += old_weights[i]
        l = old_pi0s[i] - po
        a0 = np.linalg.norm(l) + np.random.normal(0, 5)
        l /= l.item(1)
        l[0,0] += np.random.normal(0,0.001)
        l /= np.linalg.norm(l)
        pi0, vi = calculate_trajectory_first(a0, [l]+lms[1:],ec, tau)
        particle_p[mm] = pi0+vi*t
        particle_v[mm] = vi
        pi0s[mm]=pi0
    return pi0s, particle_p, particle_v

t=0.
ts = 0.2
t_s = [t, t+ts]
traj = Trajectories(particle_p, particle_v)
plotter = Plotter(initialc-1, len(traj.get_Iposes()), [[-130,70],[-5,130]])
while t < tau:
    if t > (desired_measurements-1)*ts:
        lm = pi+vi*t - po-vo*t
        lm /= lm.item(1)
        lm[0,0] += np.random.normal(0,0.0005)
        lm /= np.linalg.norm(lm)
        lms.append(lm)
        t_s.append(t)
        # weight the particles based on the new bearing measurement
        weights = update_weights(weights, lm, traj.get_Iposes()[1:], traj.get_Opos())

        # resample the particles based on the weights
        pi0s, particle_p[2:], particle_v[2:] = resample(pi0s, particle_p[2:], particle_v[2:], weights, t)
    plotter.update_plot(traj.get_Opos(),traj.get_Iposes())
    traj.update()
    t+=ts
    # time.sleep(ts)

plotter.update_plot(traj.get_Opos(), traj.get_Iposes())
plt.ioff()
plotter.plot_interactive()
plt.show()