import numpy as np
import numpy.linalg as la
from copy import deepcopy

class Particle_Filter:
    def __init__(self, num_particles, l1, l2, tau, po0, po1, ec, r_min, r_max, v_max, ts) -> None:
        self.t = ts
        self.num_particles = num_particles
        self.po0 = po0
        self.pos = [po0, po1]
        self.ts = ts
        self.Rinv = np.diag([1/0.1**2, 1/0.1**2])
        tau += ts
        self.taus = [tau]
        self.ec = ec
        self.lms = deepcopy([l1,l2])
        lms_norm = [l1/l1.item(1), l2/l2.item(1)]
        self.weights = []
        self.pi0s = []
        self.particle_p = []
        self.vis = []
        self.r_min = r_min
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
            pos, vel = self.calculate_trajectory_first(a1_u, lus, ec, tau_u, self.pos)
            if np.linalg.norm(vel)<=v_max:
                self.pi0s.append(pos)
                self.particle_p.append(pos+vel*ts)
                self.vis.append(vel)
                self.weights.append(1)
    def get_particle_positions(self):
        return self.particle_p

    def get_future_positions(self, delta_t):
        future_pos = []
        for i in range(self.num_particles):
            future_pos.append(self.particle_p[i] + self.vis[i]*delta_t)
        return future_pos
    
    # Implementation of Gauss-Newton batch discrete-time estimation (taken from State Estimation for Robotics by Tim Barfoot, pp 128-134)
    def calculate_velocity_improved(self, ls, pi0, vi, pOs):
        # vk = np.array([[1., 1.]]).T
        k = len(ls)
        R = [0.01**2]*2
        Q = [0.0001]*4
        P0 = [0.0001, 0.0001, 100., 100.]
        W = np.diag(P0 + Q*(k-1) + R*k)
        Winv = la.inv(W)
        def calc_G(x, po):
            pr = x[0:2] - po
            prx = pr.item(0)
            pry = pr.item(1)
            denom = pow(prx**2+pry**2, 3/2)
            G = np.array([[pry**2, -prx*pry, 0, 0],
                          [-prx*pry, prx**2, 0, 0]])/denom
            return G
        def g(x, po):
            pr = x[0:2] - po
            return pr/la.norm(pr)
        # def calc_G_tau(x, po):

        F = np.eye(4)
        F[0:2,2:] = self.ts*np.eye(2)
        Htop = np.eye(4*k)
        for i in range(k-1):
            Htop[4*(i+1):4*(i+2),4*i:4*(i+1)] = -F
        
        x = [deepcopy(pi0), deepcopy(vi)]
        x0 = deepcopy(x)
        x0 = np.concatenate(x0, axis=0)
        pn = deepcopy(pi0)
        for i in range(k-1):
            pn += vi*self.ts
            x.append(deepcopy(pn))
            x.append(deepcopy(vi))
        x = np.concatenate(x, axis=0)

        dx = np.ones_like(x)
        iter = 0
        amp = 0.005
        e = np.zeros((6*k,1))
        Hbottom = np.zeros((2*k,4*k))
        o = 4*k #offset for second half of error array
        while (dx.max() > amp or dx.min() < -amp):
            for i in range(k):
                if i == 0:
                    norm = x0
                else:
                    norm = F@x[4*(i-1):4*(i)]
                e[4*i:4*(i+1)] = norm - x[4*i:4*(i+1)]
                e[o+2*i:o+2*(i+1)] = ls[i] - g(x[4*i:4*(i+1)], pOs[i])
                Hbottom[2*i:2*(i+1),4*i:4*(i+1)] = calc_G(x[4*i:4*(i+1)],pOs[i])
            H = np.concatenate([Htop,Hbottom],axis=0)
            dx = la.pinv(H.T @ Winv @ H) @ H.T @ Winv @ e
            x += dx
            iter += 1

        return x[4*(k-1):4*(k-1)+2], x[4*(k-1)+2:], x[0:2] # return pk, vk, p0

    
    def calculate_trajectory_first(self, a1, ls, ec, tau, pos):
        if len(ls)<2:
            raise
        vo = (pos[-1] - pos[-2])/self.ts
        l1 = ls[0]
        ls = ls[1:]
        po0 = pos[0]
        pos = pos[1:]
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
            Ap.append(-np.eye(2)*self.ts*(i+1))
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
            b.append(po0-pos[i]+a1*l1)
        b.append(-vo*tau+a1*l1)
        b = np.concatenate(b,axis=0)

        x = np.linalg.pinv(A)@b

        # set the result up as a possible trajectory to plot along 
        # with the original example
        pi0 = a1*l1 + self.po0
        vi = x[-2:]
        return pi0, vi
    
    def update(self, lm, po, tau):
        # update the weights
        self.pos.append(deepcopy(po))
        for i in range(self.num_particles):
            # propogate the dynamics
            self.particle_p[i] += self.vis[i]*self.ts

            # get the weights based on the measurement
            phat = self.particle_p[i]-po
            phat /= np.linalg.norm(phat)
            self.weights[i] = np.exp(-1/2. * (lm-phat).T @ self.Rinv @ (lm-phat)).item(0)
        sum = np.sum(self.weights)
        self.weights = [x/sum for x in self.weights]
        self.lms.append(deepcopy(lm))
        self.t += self.ts
        self.taus.append(tau + self.t)

        # resample
        old_pi0s = deepcopy(self.pi0s)
        # old_ps = deepcopy(self.particle_p)

        rr = np.random.rand()/self.num_particles
        i = 0
        cc = self.weights[i]
        for mm in range(self.num_particles):
            u = rr + (mm-1)/self.num_particles
            while u > cc:
                i += 1
                cc += self.weights[i]
            # old way
            # l = old_pi0s[i] - self.po0
            # a0 = max(np.linalg.norm(l) + np.random.normal(0, 5), self.r_min)
            # l /= l.item(1)
            # l[0,0] += np.random.normal(0,0.001)
            # l /= np.linalg.norm(l)
            # pi0, vi = self.calculate_trajectory_first(a0, [l]+self.lms[1:],self.ec, np.average(self.taus), self.pos) # use this to get an initial guess of the position and velocity
            # pk, vk, pi0n = self.calculate_velocity_improved(self.lms, pi0+vi*self.t, vi, self.pos)

            # new way
            l = old_pi0s[i] - self.po0
            a0 = max(np.linalg.norm(l) + np.random.normal(0, 5), self.r_min)
            l /= l.item(1)
            l[0,0] += np.random.normal(0,0.001)
            l /= np.linalg.norm(l)
            vi = np.array([[1., 1.]]).T
            pi0, vi = self.calculate_trajectory_first(a0, [l]+self.lms[1:],self.ec, np.average(self.taus), self.pos) # use this to get an initial guess of the position and velocity
            # pk, vk, pi0n = self.calculate_velocity_improved(self.lms, pi0, vi, self.pos)
            pk, vk, pi0n = (pi0+vi*self.t, vi, pi0)
            self.particle_p[mm] = pk
            self.vis[mm] = vk
            self.pi0s[mm]=pi0n
