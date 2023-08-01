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

desired_measurements = 10
if desired_measurements < 2:
    raise

po=np.array([[0.,0.]]).T
vo=np.array([[0.,20.]]).T

# generate an example trajectory to calculate LOS vectors and TTC
pi=np.array([[-30.,100.]]).T
vi=np.array([[20.,0.]]).T

# generate LOS vectors from example
lms = []

ts=0.1
for i in range(desired_measurements):
    lm = pi+vi*i*ts - po-vo*i*ts
    lm /= np.linalg.norm(lm)
    lms.append(lm)

# calculate the TTC (assume the camera is facing line of travel)
ec=vo/np.linalg.norm(vo)
tau = ((po-pi).T @ ec)/((vi-vo).T @ ec)

def calculate_trajectory(a1, ls, ec, tau):
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

# def calculate_positions_and_velocities(l1, l2, ec, tau):

#     # get the unit vector perpendicular to the camera normal vector
#     ecp = np.array([[0, -1],[1,0]]) @ ec

#     # solve the min-norm problem
#     A1 = np.concatenate([l1,-l2,np.zeros((2,1)),np.eye(2)*ts],axis=1)
#     A2 = np.concatenate([l1,np.zeros((2,1)),-ecp,np.eye(2)*tau], axis=1)
#     A = np.concatenate([A1,A2],axis=0)

#     b = np.concatenate([vo*ts,vo*tau],axis=0)

#     x = np.linalg.pinv(A)@b

#     # set the result up as a possible trajectory to plot along 
#     # with the original example
#     vi2 = x[3:]
#     a1 = x.item(0)
#     pi2 = l1*a1+po

#     # run an optimizer to get the closest and furthest starting 
#     # points possible while respecting the constraints above

#     # function to minimize alpha
#     def f_min(x):
#         return x
#     # function to maximize alpha
#     def f_max(x):
#         return -x
#     # function for maximum velocity constraint
#     def max_vel(x):
#         vel = x*(vi2-vo)+vo
#         return v_max - np.linalg.norm(vel)
#     # function for maximum range constraint
#     r0 = np.linalg.norm((pi2-po))
#     def max_range(x):
#         return r_max - x*r0
#     # function for minimum range constraint
#     def min_range(x):
#         return x*r0 - r_min

#     cons = ({'type':'ineq', 'fun':max_vel}, 
#             {'type':'ineq', 'fun':max_range},
#             {'type':'ineq', 'fun':min_range})

#     x0=1.

#     # do the optimization to minimize alpha
#     result = minimize(f_min,x0, constraints=cons)
#     p_min = result.x.item(0)*(pi2-po)+po
#     pi2t = result.x.item(0)*((pi2+vi2*tau)-(po+vo*tau))+(po+vo*tau)
#     vt_min = (pi2t-p_min)/tau

#     # do the optimization to maximize alpha
#     result = minimize(f_max, x0, constraints=cons)
#     p_max = result.x.item(0)*(pi2-po)+po
#     pi2t = result.x.item(0)*((pi2+vi2*tau)-(po+vo*tau))+(po+vo*tau)
#     vt_max = (pi2t-p_max)/tau
#     return (p_min, vt_min, p_max,vt_max)

pairs = []
labels = []



# res = calculate_positions_and_velocities(l1, l2, ec, tau)
# pairs.append(res)
# labels.append("Avg")

# # rerun the calculations for l1 being slightly off
# l1_minus = l1 / l1.item(1)
# l1_minus[0,:]-=0.01
# l1_minus /= np.linalg.norm(l1_minus)
# print("Angle Difference (minus) (degrees):")
# print(np.arccos(l1.T @ l1_minus)*180/np.pi)
# print()
# res1m = calculate_positions_and_velocities(l1_minus, l2, ec, tau)
# pairs.append(res1m)
# labels.append("1 Minus")

# l1_plus = l1/l1.item(1)
# l1_plus[0,:] += 0.01
# l1_plus /= np.linalg.norm(l1_plus)
# print("Angle Difference (plus) (degrees):")
# print(np.arccos(l1.T@l1_plus)*180/np.pi)
# print()
# res1p = calculate_positions_and_velocities(l1_plus, l2, ec, tau)
# pairs.append(res1p)
# labels.append("1 Plus")

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

        self.ax.scatter(self.plot_x, self.plot_y,s=1)

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
        samp = Slider(axamp, 'Amp', 0, len(self.pox)-1, valinit=initial_i, valstep=1)

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

num_particles = 1000
lms_norm = []
for lm in lms:
    lm_norm = lm / lm.item(1)
    lms_norm.append(lm_norm)
particle_p = [po, pi]
particle_v = [vo, vi]
initialc = len(particle_p)
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
    pos, vel = calculate_trajectory(a1_u, lus, ec, tau_u)
    if np.linalg.norm(vel)<=v_max:
        particle_p.append(pos)
        particle_v.append(vel)


t=0.
ts = 0.2
traj = Trajectories(particle_p, particle_v)
plotter = Plotter(initialc-1, len(traj.get_Iposes()), [[-130,70],[-5,130]])
while t < tau:
    plotter.update_plot(traj.get_Opos(),traj.get_Iposes())
    traj.update()
    t+=ts
    # time.sleep(ts)

plotter.update_plot(traj.get_Opos(), traj.get_Iposes())
plt.ioff()
plotter.plot_interactive()
plt.show()