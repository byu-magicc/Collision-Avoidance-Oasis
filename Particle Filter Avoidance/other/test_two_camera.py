import numpy as np


po=np.array([[0.,0.]]).T
vo=np.array([[0.,20.]]).T

# generate an example trajectory to calculate LOS vectors and TTC
pi=np.array([[-5.,100.]]).T
vi=np.array([[-5.,-20.]]).T

# generate LOS vectors from example
l1=pi-po
l1/=np.linalg.norm(l1)

ts=0.1
l2 = pi+vi*ts - po-vo*ts
l2 /= np.linalg.norm(l2)

# calculate the TTC (assume the camera is facing line of travel)
ec1=vo/np.linalg.norm(vo)
tau1 = ((po-pi).T @ ec1)/((vi-vo).T @ ec1)
# print(tau1)

ec2 = np.array([[1.,1.]]).T
ec2 /= np.linalg.norm(ec2)
tau2 = ((po-pi).T @ ec2)/((vi-vo).T @ ec2)
# print(tau2)

# get the unit vector perpendicular to the camera normal vector
ecp1 = np.array([[0, -1],[1,0]]) @ ec1
ecp2 = np.array([[0, -1],[1,0]]) @ ec2

# solve the min-norm problem
zero = np.zeros((2,1))
A1 = np.concatenate([l1,-l2,zero, zero,np.eye(2)*ts],axis=1)
A2 = np.concatenate([l1,zero,-ecp1, zero,np.eye(2)*tau1], axis=1)
A3 = np.concatenate([l1,zero, zero,-ecp2, np.eye(2)*tau2], axis=1)
A = np.concatenate([A1,A2, A3],axis=0)

b = np.concatenate([vo*ts,vo*tau1, vo*tau2],axis=0)

x = np.linalg.inv(A)@b
print("Inverse:")
print(x)

print()
print("Pseudo-Inverse:")
print(np.linalg.pinv(A)@b)


print()
print(A)
print()
print("Rank of A:")
print(np.linalg.matrix_rank(A))