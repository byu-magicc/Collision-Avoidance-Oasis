import numpy as np
import matplotlib.pyplot as plt
import time

limits=[[-130,70],[-5,130]]

po=np.array([[0.,0.]]).T
vo=np.array([[0.,20.]]).T

pi=np.array([[-30.,100.]]).T
vi=np.array([[0.,-20.]]).T

alpha = 1.2
pi2 = alpha*(pi-po)+po

ec = vo/np.linalg.norm(vo)
tau = ((po-pi).T @ ec)/((vi-vo).T @ ec)

pi2t = alpha*((pi+vi*tau)-(po+vo*tau))+(po+vo*tau)
vi2 = (pi2t-pi2)/tau

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
    def __init__(self, num_intruders) -> None:
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.num_intruders = num_intruders
        self.pox = []
        self.poy = []
        self.pix = []
        self.piy = []
        for i in range(num_intruders):
            self.pix.append([])
            self.piy.append([])

    def update_plot(self, own_pos, i_poses):
        self.ax.clear()
        self.pox.append(own_pos.item(0))
        self.poy.append(own_pos.item(1))
        self.ax.plot(self.pox,self.poy,label='Own')
        for i in range(self.num_intruders):
            self.pix[i].append(i_poses[i].item(0))
            self.piy[i].append(i_poses[i].item(1))
            self.ax.plot(self.pix[i],self.piy[i],label=f"Intruder {i+1}")

        for i in range(self.num_intruders):
            self.ax.plot([self.pox[-1],self.pix[i][-1]],[self.poy[-1],self.piy[i][-1]],label=f"Bearing {i+1}")

        self.ax.set_title("Positions of Own-ship and Intruders")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.legend()
        self.ax.set_xlim(limits[0])
        self.ax.set_ylim(limits[1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

t=0.
ts = 0.5
traj = Trajectories([po, pi, pi2], [vo, vi, vi2])
plotter = Plotter(2)
while t < tau:
    plotter.update_plot(traj.get_Opos(),traj.get_Iposes())
    traj.update()
    t+=ts
    time.sleep(ts)

plt.ioff()
plotter.update_plot(traj.get_Opos(), traj.get_Iposes())
plt.show()