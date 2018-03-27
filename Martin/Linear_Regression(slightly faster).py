# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 13:13:09 2018

@author: msed
"""

import numpy as np
import matplotlib.pyplot as plt

""" Trying to implement Algorithm 1 out of the paper Zhang et al. for linear regression"""
def lin_reg(data_x,data_y,epsilon):
    #input pairs (x,y) and privacy budget epsilon; each line of data_x is supposed to be one input vector
    #assume both data_x and data_y are numpy arrays
    (n,d)=data_x.shape
    print(n,d)
    #assuming data_x and data_y don't have absolute value greater than 1
    Delta=2*(1+2*d+d^2)
    # j = 0 : constant terms (in w) in objective function
    lambd_0=(data_y**2).sum()+np.random.laplace(scale=Delta/epsilon)
    # j = 1 : linear terms (in w) in objective function
    # lambd_1 is a d dimensional array holding all the coefficients for the d monomials in w
    lambd_1=-2*data_y.dot(data_x)+np.random.laplace(scale=Delta/epsilon,size=d)
    # j = 2: quadratic terms (in w) in objective function
    # lambd_2 is a d x d dimensional array holding all the coefficients for the d x d quadratic terms of form w_j*w_l
    lambd_2=data_x.transpose().dot(data_x)+np.random.laplace(scale=Delta/epsilon)
    
    lambd_2_new=lambd_2+np.transpose(lambd_2)
    #solve quadratic optimization problem: minimize lambd_0+lambd_1^T*w+1/2w^T*lambd_2_new*w
    #==> solve lambd_2_new * w = -lambd_1
    w=np.linalg.solve(lambd_2_new,-lambd_1)
    return(w)

"""small example to show that in case of very big epsilon (noise effectively zero) we get correct linear regression"""

data_x=np.array([[1,0.2],[1,0.4],[1,0.6]]) # add a x_0 = 1 coordinate for computing w_0
data_y=np.array([0.1,0.5,0.6])

x1=np.linspace(0,1,20)
w=lin_reg(data_x,data_y,1000)


def lin(w, x):
    return w[1]*x+w[0]

print(lin(w,x1))

plt.scatter(data_x[:,1],data_y,color='red')
plt.plot(x1,lin(w,x1),color='blue',linestyle='--')
plt.show()