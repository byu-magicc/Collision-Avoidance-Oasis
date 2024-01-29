"""
Optimization algorithm deisnged to optimize the desired unit vectors
for a target approaching the UAV
"""
import numpy as np
import matplotlib.pyplot as plt
import time

sigma = 0.01

n = 10 # number of points to produce

#starting and ending radii
start_rad = 100.
end_rad = 50.

# setup the repulsive plane
d_point = [np.array([[-1,0,0]]).T, np.array([[0,-1,0]]).T]
norm_p = [np.array([[1,0,0]]).T, np.array([[0,1,0]]).T]

# setup constants for the optimization
k_m = 5
k_g = 1
k_b = [5, 2]

# fix the first point
first_lam = np.array([[1,0,0]]).T
lams = [first_lam]

#generate a series of other lambdas of unit length in the positive x half-plane
for i in range(n-1):
    next = np.array([[np.random.rand(), 2*np.random.rand()-1, 2*np.random.rand()-1]]).T
    next /= np.linalg.norm(next)
    lams.append(next)

# generate a series of radii to use in calculation
rad = np.linspace(start_rad, end_rad, n)


def compute_G(lams):
    G = np.zeros((3,3))
    for i in range(n):
        G += 1/(sigma*rad[i])**2*lams[i] @ lams[i].T
    return G
def compute_P(lam):
    return np.eye(3) - lam@lam.T

iterations = 1500
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def plot_lams(lams, iter):
    # add each lambda to a x, y and z list of values for plotting
    x=[]
    y=[]
    z=[]
    for lam in lams:
        x.append(lam.item(0))
        y.append(lam.item(1))
        z.append(lam.item(2))
    ax.clear()
    ax.scatter(x,y,zs=z, zdir='z', label="Final lambda positions")

    ax.set_xbound(-1,1)
    ax.set_ybound(-1,1)
    ax.set_zbound(-1,1)

    ax.set_title(f"Iteration {iter}")
    ax.set_xlabel("x")
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    fig.canvas.draw()
    fig.canvas.flush_events()

for iter in range(iterations):
    G = compute_G(lams)
    new_lams = [lams[0]]
    for i in range(1,n): #compute the pseudo-forces for each particle (except the first one)
        P = compute_P(lams[i])
        force = -k_g*P @ G @ lams[i]
        force += k_m * P @ (lams[i-1]-lams[i])
        for j in range(len(norm_p)):
            distance_to_plane = np.abs(norm_p[j].T @ (lams[i]-d_point[j]))
            force += k_b[j] * P @ norm_p[j]/distance_to_plane
        gain=0.001
        #apply the pseudoforce to the particle
        new_lams.append(lams[i]+gain*force)
        new_lams[i] /= np.linalg.norm(new_lams[i])
    lams=new_lams
    plot_lams(lams, iter)
    time.sleep(0.01)


# add each lambda to a x, y and z list of values for plotting
x=[]
y=[]
z=[]
for lam in lams:
    x.append(lam.item(0))
    y.append(lam.item(1))
    z.append(lam.item(2))
plt.ioff()
ax.clear()
ax.set_title(f"Iteration {iterations}")
ax.scatter(x,y,zs=z, zdir='z', label="Final lambda positions")
ax.set_xbound(-1,1)
ax.set_ybound(-1,1)
ax.set_zbound(-1,1)
ax.set_xlabel("x")
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()