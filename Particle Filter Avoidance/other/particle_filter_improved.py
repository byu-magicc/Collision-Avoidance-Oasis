import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Ellipse
from particle_filter import Particle_Filter
import time
from copy import deepcopy
from path_planner import PathPlanner

# define constraints for the optimizer to use later on
# minimum and maximum ranges of detection
r_min = 10
r_max = 1000

# maximum predicted velocity (90m/s~200mph)
v_max = 90

# maximum own-ship velocity (23m/s~50mph)
vo_max = 23

po=np.array([[0.,0.]]).T
vo=np.array([[0.,20.]]).T

# generate an example trajectory to calculate LOS vectors and TTC
actual_pis=[np.array([[-100.,100.]]).T, np.array([[30., 50.]]).T]
actual_vis=[np.array([[20.,0.]]).T, np.array([[-20., 0.]]).T]

ec=vo/np.linalg.norm(vo)


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
    
    def set_own_position(self, ownp):
        self.poss[0] = ownp

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
        plt.ion()
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
            lines=self.ax.plot(self.pix[i],self.piy[i], marker='.', markersize=10,label=f"Intruder {i+1}")
            if len(self.pix[i]) >= 2:
                self.ax.arrow(self.pix[i][0], self.piy[i][0], self.pix[i][1]-self.pix[i][0], self.piy[i][1]-self.piy[i][0], head_width=head_width, color=lines[0].get_color())
            

        # plot the own-ship path
        self.ax.plot(self.pox,self.poy, marker='.', markersize=10,label='Own',c='r')
        if len(self.pox)>=2:
            self.ax.arrow(self.pox[0], self.poy[0], self.pox[1]-self.pox[0], self.poy[1]-self.poy[0], color='r', head_width=head_width)

        self.ax.set_title("Particles Produced By Intruders")
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

planner = PathPlanner((0,200))
from scipy.stats import gaussian_kde
def calculate_problematic_and_pdfs(t, ts, filters, po, vo, vo_max):
    max_dt = 5.
    kdes = []
    problematic = []
    not_problematic = []

    nf = len(filters)
    dts = np.arange(0, max_dt+ts, ts)
    for i in range(nf):
        kdes.append([])
        problematic.append([])
        not_problematic.append([])
        for j in range(len(dts)):
            problematic[i].append([])
            not_problematic[i].append([])
    for i, dt in enumerate(dts):
        for j in range(nf):
            x = []
            y = []
            pos = filters[j].get_future_positions(dt)
            for p in pos:
                # if la.norm(p-po-vo*t)/(dt+0.0001)<=vo_max:
                problematic[j][i].append(p)
                    
                # else:
                #     not_problematic[j][i].append(p)
                x.append(p.item(0))
                y.append(p.item(1))
            if len(problematic[j][i]) <= 2:
                kdes[j].append(None)
            else:
                x = np.array(x)
                y = np.array(y)
                k = gaussian_kde(np.array([x,y]))
                kdes[j].append(k)
    return kdes, problematic, not_problematic

            


def plot_futures(t, ts, filters, actual_pis, actual_vis, po, vo, xlims, ylims):
    plt.ioff()
    fig, ax = plt.subplots()

    initial_dt = 0
    max_dt = 5.
    problematic_particle_plots = []
    not_problematic_particle_plots = []
    intruder_plots = []
    p_ellipses = []
    ax_gca = plt.gca()
    contours = [None]*len(filters)

    kdes, problematic, not_problematic = calculate_problematic_and_pdfs(t, ts, filters, po, vo, vo_max)
    if t > 1:
        path, cps = planner.update(po, kdes)
    tf = t
    j = int((tf - t)/ts)

    for i in range(len(filters)):
        x = [p.item(0) for p in problematic[i][j]]
        y = [p.item(1) for p in problematic[i][j]]
        l,=plt.plot(x, y,marker='.', ls='', markersize=1, label=f'Intruder {i+1} Particles')

        # nl,=plt.plot([p.item(0) for p in not_problematic[i][j]], [p.item(1) for p in not_problematic[i][j]],marker='.', ls='', markersize=1, label=f'Not P Particles Intruder {i+1}')
        if contours[i] is not None:
            for coll in contours[i].collections:
                coll.remove()
            contours[i] = None
        # if len(problematic[i][j]) > 1 and kdes[i][j] is not None:
            # centroid, a, b, alpha = get_ellipse_for_printing(problematic[i][j])
        # x = np.array(x)
        # y = np.array(y)
        # # k = kdes[i][j]
        # nbins = 100

        # xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        # zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # contours[i] = ax.contour(xi, yi, zi.reshape(xi.shape))
        # else:
        centroid, a, b, alpha = (0,-100),5.,5.,0.
            
        ellipse = Ellipse(centroid, a, b, alpha, edgecolor='k', fc='None',lw=2)
        ax_gca.add_artist(ellipse)
        p_ellipses.append(ellipse)
        problematic_particle_plots.append(l)
        # not_problematic_particle_plots.append(nl)
    for i in range(len(actual_pis)):
        p = actual_pis[i]+actual_vis[i]*(t+initial_dt)
        li, = plt.plot(p.item(0), p.item(1), marker='.', ls='', markersize=10, label=f'Intruder {i+1}')
        intruder_plots.append(li)
    p = po+vo*(initial_dt)
    if t <= 1:
        l0, = plt.plot(p.item(0), p.item(1), c='r', marker='.', ls='', markersize=10, label='Ownship')
    else:
        l0, = plt.plot(cps[j][0], cps[j][1], c='r', marker='.', ls='', markersize=10, label='Ownship')
        curvepts = np.array(path.evalpts)
        curveplt, = plt.plot(curvepts[:,0], curvepts[:,1], color='yellowgreen', linestyle='-')
    ax_gca = plt.axis([xlims[0], xlims[1], ylims[0], ylims[1]])
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.title(f"Future Predictions for t={t:.3f}s")

    axamp = plt.axes([0.25, .03, 0.50, 0.02])
    # Slider
    samp = Slider(axamp, 'Time', t, t+max_dt, valinit=initial_dt, valstep=ts)

    def update(val):
        # amp is the current value of the slider
        tf = samp.val
        j = int((tf - t)/ts)
        # update curve
        for i in range(len(filters)):
            x = [p.item(0) for p in problematic[i][j]]
            y = [p.item(1) for p in problematic[i][j]]
            problematic_particle_plots[i].set_xdata(x)
            problematic_particle_plots[i].set_ydata(y)
            # not_problematic_particle_plots[i].set_xdata([p.item(0) for p in not_problematic[i][j]])
            # not_problematic_particle_plots[i].set_ydata([p.item(1) for p in not_problematic[i][j]])
            if contours[i] is not None:
                for coll in contours[i].collections:
                    coll.remove()
                contours[i] = None
            # if len(problematic[i][j]) > 1 and kdes[i][j] is not None:
                # centroid, a, b, alpha = get_ellipse_for_printing(problematic[i][j])
                # x = np.array(x)
                # y = np.array(y)
                # k = kdes[i][j]
                # nbins = 100

                # xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
                # zi = k(np.vstack([xi.flatten(), yi.flatten()]))

                # contours[i] = ax.contour(xi, yi, zi.reshape(xi.shape))
            # else:
            centroid, a, b, alpha = (0,-100),5.,5.,0.
            p_ellipses[i].set_center(centroid)
            p_ellipses[i].set_width(a)
            p_ellipses[i].set_height(b)
            p_ellipses[i].set_angle(alpha)
            p = actual_pis[i]+actual_vis[i]*(tf)
            intruder_plots[i].set_xdata(p.item(0))
            intruder_plots[i].set_ydata(p.item(1))
        
        if t <= 1:
            p = po+vo*(j*ts)
            l0.set_xdata(p.item(0))
            l0.set_ydata(p.item(1))
        else:
            l0.set_xdata(cps[j][0])
            l0.set_ydata(cps[j][1])
        # redraw canvas while idle
        fig.canvas.draw()

    # call update function on slider value change
    samp.on_changed(update)

    plt.show()
    plt.ion()
    # return np.array([[cps[1][0], cps[1][1]]]).T

def get_ellipse_for_printing(points):
    center = np.mean(np.asarray(points), axis=0)
    reshaped = np.reshape(np.stack(points, axis=1),(2,-1))
    A = np.cov(reshaped)#np.zeros((center.shape[0],center.shape[0]))
    # n = len(points)
    # for point in points:
    #     A += (point - center) @ (point - center).T/(n-1)
    
    # print A

    # V is the rotation matrix that gives the orientation of the ellipsoid.
    # https://en.wikipedia.org/wiki/Rotation_matrix
    # http://mathworld.wolfram.com/RotationMatrix.html
    U, D, V = la.svd(A)
    
    # x, y radii.
    l1, l2 = np.sqrt(D)
    # Major and minor semi-axis of the ellipse.
    s = np.sqrt(5.991) # cooresponds to 95% confidence interval
    dx, dy = 2 * l1 * s, 2 * l2 * s
    a, b = max(dx, dy), min(dx, dy)
    # Eccentricity
    e = np.sqrt(a ** 2 - b ** 2) / a

    # print '\n', U
    # print D
    # print V, '\n'
    # Orientation angle (with respect to the x axis counterclockwise).
    alpha = np.rad2deg(np.arctan2(V[0][1],V[0][0]))
    centroid = (center.item(0), center.item(1))
    return centroid, a, b, alpha

num_particles = 1000
particle_p = deepcopy([po]+actual_pis)
particle_v = [vo]+actual_vis

t=0.
ts = 0.2
tstop = 5.
steps = 0
num_intruders = len(actual_pis)
lm_col = []
filters = []
for i in range(num_intruders):
    lm_col.append([])

traj = Trajectories(num_intruders, num_particles, particle_p, particle_v, ts)
plotter = Plotter(num_intruders, num_particles, [[-130,70],[-5,130]])
following_path = False
while t < tstop:
    for i in range(num_intruders):
        lm = traj.get_intruder_positions()[i] - traj.get_own_position()
        # corrupt the bearing measurement with noise
        lm /= lm.item(1)
        lm[0,0] += np.random.normal(0,0.0005)
        lm /= np.linalg.norm(lm)
        lm_col[i].append(lm)
        # calculate tau
        tau = ((traj.get_own_position()-traj.get_intruder_positions()[i]).T @ ec)/((actual_vis[i]-vo).T @ ec)
        if steps == 1:
            # initialize the filters
            filters.append(Particle_Filter(num_particles, lm_col[i][0], lm_col[i][1], tau, po, po+vo*ts, ec, r_min, r_max, v_max, ts))
        if steps >= 2:
            # weight the particles based on the new bearing measurement
            filters[i].update(lm, traj.get_own_position(), tau, following_path)
    if steps >= 1:
        plotter.update_plot(traj.get_own_position(), traj.get_intruder_positions(), [filter.get_particle_positions() for filter in filters])
        nextpoint = plot_futures(t, ts, filters, actual_pis, actual_vis, traj.get_own_position(), vo, [-200, 200], [0, 200])
    traj.update()
    # if steps >= 1:
    #     traj.set_own_position(nextpoint)
    t+=ts
    steps += 1
    # time.sleep(ts)

plotter.update_plot(traj.get_own_position(), traj.get_intruder_positions(), [filter.get_particle_positions() for filter in filters])
plotter.plot_interactive()
plt.show()