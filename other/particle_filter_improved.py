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
    def __init__(self, num_intruders, num_particles, initial_pos, velocities, ts) -> None:
        self.poss = deepcopy(initial_pos)
        self.vels = deepcopy(velocities)
        self.num_intruders = num_intruders
        self.num_particles = num_particles
        self.ts = ts

    def update(self):
        for i in range(len(self.poss)):
            self.poss[i] += self.vels[i]*self.ts
    
    def get_own_position(self):
        return self.poss[0]
    
    def get_intruder_positions(self):
        return self.poss[1:self.num_intruders+1]
    
    def get_particle_positions(self):
        particleps = []
        for i in range(self.num_intruders):
            particleps.append(self.poss[self.num_intruders+1+i*self.num_particles:self.num_intruders+1+(i+1)*self.num_particles])
        return particleps

    def set_particle_positions(self, pposes):
        for i in range(self.num_intruders):
            self.poss[self.num_intruders+1+i*self.num_particles:self.num_intruders+1+(i+1)*self.num_particles] = pposes[i]
class Plotter:
    def __init__(self, num_intruders, num_particles, limits) -> None:
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.num_intruders = num_intruders
        self.num_particles = num_particles
        self.limits = limits
        # list of the x and y coordinates of the ownship
        self.pox = []
        self.poy = []
        # list of list of x and y coordinates of each of the intruders
        self.pix = []
        self.piy = []
        self.particles_x = []
        self.particles_y = []
        for i in range(num_intruders):
            self.pix.append([])
            self.piy.append([])
            self.particles_x.append([])
            self.particles_y.append([])

    def update_plot(self, own_pos, intruder_poses, particle_poses):
        head_width = 10
        self.ax.clear()
        # add the points to the respective lists
        self.pox.append(own_pos.item(0))
        self.poy.append(own_pos.item(1))
        for i in range(self.num_intruders):
            # add each intruder position to the list
            self.pix[i].append(intruder_poses[i].item(0))
            self.piy[i].append(intruder_poses[i].item(1))

            # add the particle positions to the lists
            for j in range(self.num_particles):
                self.particles_x[i].append(particle_poses[i][j].item(0))
                self.particles_y[i].append(particle_poses[i][j].item(1))
            # plot the particles of each intruder
            self.ax.plot(self.particles_x[i], self.particles_y[i], marker='.', ls='', markersize=1, zorder=-30, label=f"In. {i+1} Particles")

        # plot each of the actual intruders
        for i in range(self.num_intruders):
            lines=self.ax.plot(self.pix[i],self.piy[i],label=f"Intruder {i+1}")
            if len(self.pix[i]) >= 2:
                self.ax.arrow(self.pix[i][0], self.piy[i][0], self.pix[i][1]-self.pix[i][0], self.piy[i][1]-self.piy[i][0], head_width=head_width, color=lines[0].get_color())
            

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
        plt.ioff()
        fig, ax = plt.subplots()

        initial_i = 0
        particle_plots = []
        intruder_plots = []
        for i in range(self.num_intruders):
            l,=plt.plot(self.particles_x[i][0:self.num_particles], self.particles_x[i][0:self.num_particles],marker='.', ls='', markersize=1, label=f'Particles Intruder {i+1}')
            particle_plots.append(l)
        for i in range(self.num_intruders):
            li, = plt.plot(self.pix[i][0], self.piy[i][0], marker='.', ls='', markersize=10, label=f'Intruder {i+1}')
            intruder_plots.append(li)
        l0, = plt.plot(self.pox[0], self.poy[0], c='r', marker='.', ls='', markersize=10, label='Ownship')
        xlims = self.ax.get_xlim()
        ylims = self.ax.get_ylim()
        ax = plt.axis([xlims[0], xlims[1], ylims[0], ylims[1]])
        plt.legend()

        axamp = plt.axes([0.25, .03, 0.50, 0.02])
        # Slider
        samp = Slider(axamp, 'Timestep', 0, len(self.pox)-1, valinit=initial_i, valstep=1)

        def update(val):
            # amp is the current value of the slider
            j = samp.val
            # update curve
            for i in range(self.num_intruders):
                particle_plots[i].set_xdata(self.particles_x[i][self.num_particles*j:self.num_particles*(j+1)])
                particle_plots[i].set_ydata(self.particles_y[i][self.num_particles*j:self.num_particles*(j+1)])
                intruder_plots[i].set_xdata(self.pix[i][j])
                intruder_plots[i].set_ydata(self.piy[i][j])
            l0.set_xdata(self.pox[j])
            l0.set_ydata(self.poy[j])
            # redraw canvas while idle
            fig.canvas.draw_idle()

        # call update function on slider value change
        samp.on_changed(update)

        plt.show()

class Particle_Filter:
    def __init__(self, num_particles, l1, l2, tau, po0, ec, r_min, r_max, v_max, ts) -> None:
        self.num_particles = num_particles
        self.po0 = po0
        self.ts = ts
        self.Rinv = np.diag([1/0.05**2, 1/.05**2])
        self.tau = tau
        self.ec = ec
        self.lms = deepcopy([l1,l2])
        lms_norm = [l1/l1.item(1), l2/l2.item(1)]
        self.weights = []
        self.pi0s = []
        self.particle_p = []
        self.vis = []
        while len(self.pi0s) < num_particles:
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
                self.pi0s.append(pos)
                self.particle_p.append(pos)
                self.vis.append(vel)
                self.weights.append(1)
    def get_particle_positions(self):
        return self.particle_p
    
    def update_weights(self, lm, po):
        for i in range(self.num_particles):
            # propogate the dynamics
            self.particle_p[i] += self.vis[i]*self.ts

            # get the weights based on the measurement
            phat = self.particle_p[i]-po
            phat /= np.linalg.norm(phat)
            self.weights[i] = np.exp(-1/2. * (lm-phat).T @ self.Rinv @ (lm-phat)).item(0)
        sum = np.sum(self.weights)
        self.weights = [x/sum for x in self.weights]
        self.lms.append(lm)

    def resample(self, t):
        old_pi0s = deepcopy(self.pi0s)

        rr = np.random.rand()/self.num_particles
        i = 0
        cc = self.weights[i]
        for mm in range(self.num_particles):
            u = rr + (mm-1)/self.num_particles
            while u > cc:
                i += 1
                cc += self.weights[i]
            l = old_pi0s[i] - self.po0
            a0 = np.linalg.norm(l) + np.random.normal(0, 5)
            l /= l.item(1)
            l[0,0] += np.random.normal(0,0.001)
            l /= np.linalg.norm(l)
            pi0, vi = calculate_trajectory_first(a0, [l]+self.lms[1:],ec, tau)
            self.particle_p[mm] = pi0+vi*t
            self.vis[mm] = vi
            self.pi0s[mm]=pi0

num_particles = 1000
filter = Particle_Filter(num_particles, lms[0], lms[1], tau, po, ec, r_min, r_max, v_max, ts)
particle_p = deepcopy([po, pi])
particle_v = [vo, vi]

t=0.
ts = 0.2
num_intruders = 1

traj = Trajectories(num_intruders, num_particles, particle_p, particle_v, ts)
plotter = Plotter(num_intruders, num_particles, [[-130,70],[-5,130]])
while t < tau:
    if t > (desired_measurements-1)*ts:
        lm = traj.get_intruder_positions()[0] - traj.get_own_position()
        # corrupt the bearing measurement with noise
        lm /= lm.item(1)
        lm[0,0] += np.random.normal(0,0.0005)
        lm /= np.linalg.norm(lm)
        lms.append(lm)
        # weight the particles based on the new bearing measurement
        filter.update_weights(lm, traj.get_own_position())


        # resample the particles based on the weights
        filter.resample(t)
    plotter.update_plot(traj.get_own_position(), traj.get_intruder_positions(), [filter.get_particle_positions()])
    traj.update()
    t+=ts
    # time.sleep(ts)

plotter.update_plot(traj.get_own_position(), traj.get_intruder_positions(), [filter.get_particle_positions()])
plotter.plot_interactive()
plt.show()