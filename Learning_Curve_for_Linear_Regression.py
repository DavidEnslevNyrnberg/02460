# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 12:13:05 2018

@author: marti
"""
import numpy as np
import matplotlib.pyplot as plt

def lin_reg(data_x,data_y,epsilon):
    #input pairs (x,y) and privacy budget epsilon; each line of data_x is supposed to be one input vector
    #assume both data_x and data_y are numpy arrays
    (n,d)=data_x.shape
    
    #assuming data_x and data_y don't have absolute value greater than 1
    Delta=2*(1+2*d+d^2)
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

def lin(w, x):
    return w[1]*x+w[0]
    
#data_x = np.array([[1,0.2],[1,0.4],[1,0.6]]) # add a x_0 = 1 coordinate for computing w_0
#data_y = np.array([0.1,0.5,0.6])

Nmin = 5
Nmax = 200
Q = 100
Error1 = np.zeros(Nmax-Nmin)
MeanError1 = np.zeros(Nmax-Nmin)
Error2 = np.zeros(Nmax-Nmin)
MeanError2 = np.zeros(Nmax-Nmin)
Error3 = np.zeros(Nmax-Nmin)
MeanError3 = np.zeros(Nmax-Nmin)
data_x = np.zeros((Nmax,2))
data_x[:,0] = np.ones((Nmax))
data_x[:,1] = np.linspace(0,1,Nmax)
data_y = data_x[:,1]*10+5+np.random.rand(Nmax)

for j in range(Q):
    
    for n in range(Nmin,Nmax):
        w = lin_reg(data_x[0:n],data_y[0:n],10000)
        Error1[n-Nmin] = np.mean((lin(w,data_x[:,1])-data_y)**2) 
        w = lin_reg(data_x[0:n],data_y[0:n],1000)
        Error2[n-Nmin] = np.mean((lin(w,data_x[:,1])-data_y)**2) 
        w = lin_reg(data_x[0:n],data_y[0:n],100)
        Error3[n-Nmin] = np.mean((lin(w,data_x[:,1])-data_y)**2) 
    MeanError1 = MeanError1+Error1
    MeanError2 = MeanError2+Error2
    MeanError3 = MeanError3+Error3
MeanError1 = MeanError1/Q
MeanError2 = MeanError2/Q
MeanError3 = MeanError3/Q

#plt.plot(np.linspace(Nmin,Nmax,Nmax-Nmin),MeanError1,color='red')
#plt.plot(np.linspace(Nmin,Nmax,Nmax-Nmin),MeanError2,color='blue',linestyle='--')
plot1, = plt.plot(np.linspace(Nmin,Nmax,Nmax-Nmin),MeanError1,color='red', label='eps=10000')
plot2, = plt.plot(np.linspace(Nmin,Nmax,Nmax-Nmin),MeanError2,color='blue',linestyle='--', label='eps=1000')
plot3, = plt.plot(np.linspace(Nmin,Nmax,Nmax-Nmin),MeanError3,color='green',linestyle='-.', label='eps=100')
plt.legend(handles=[plot1, plot2, plot3])
plt.axis([Nmin,Nmax,0,1]) # Try to remove this
plt.show()



#plt.scatter(data_x[:,1],data_y,color='red')
#plt.plot(data_x[0:n],lin(w,data_x[0:n]),color='blue',linestyle='--')
#plt.show()


