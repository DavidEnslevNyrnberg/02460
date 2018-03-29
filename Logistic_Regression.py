from scipy.optimize import minimize
import numpy as np
import math
import matplotlib.pyplot as plt
""" The logistic regression algorithm as given by Chaudhuri, Monteleoni 2008"""
def private_logistic_regression(lam,data_x,data_y,epsilon, num_steps, learning_rate):
    # input pairs (x,y) and privacy budget epsilon; each line of data_x is supposed to be one input vector
    # assume both data_x and data_y are numpy arrays
    # lambda is the regularization parameter
    (n, d) = data_x.shape
    #first, draw a noise vector b distributed like exp(-eps/2*||b||)
    #to do this, first pick the norm according to gamma distribution:
    b_norm=np.random.gamma(d, scale=1/epsilon)
    print("b_norm"+str(b_norm))
    #b_norm=0
    #then direction randomly in d-dimensional space (http://mathworld.wolfram.com/HyperspherePointPicking.html)
    bx=np.random.normal(size=d)
    b=bx/np.linalg.norm(bx)*b_norm
    #now find minimizer of the objective function (given below)
    w_initial=np.zeros(d)
    w=w_initial
    step=0
    w_gradient=np.ones(d)
    while (np.linalg.norm(w_gradient)>10**(-10)) & (step<num_steps):
        # Update weights with gradient descent
        w_gradient=gradient(lam,data_x,data_y,w,b)
        w-=learning_rate*w_gradient
        if step%1000==0:
            print(step)
        step+=1
    return w

def gradient(lam,data_x,data_y,w,b):
    (n,d)=data_x.shape
    return (lam*w+b/n-1/n*data_x.T.dot(np.multiply(data_y,1/(1+np.exp(np.multiply(data_y,(data_x.dot(w))))))))
    #return(lam*w+b/n+1/n*sum([-data_y[i]*data_x[i,:]*1/(1+math.exp(data_y[i]*w.dot(data_x[i,:]))) for i in range(n)]))

np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
#our algorithm assumes the labels are in {-1,1}
simulated_labels = np.hstack((-np.ones(num_observations),np.ones(num_observations)))
#sklearn's algorithm assumes the labels are in {0,1}
simulated_labels_2=np.hstack((np.zeros(num_observations),np.ones(num_observations)))

plt.figure(figsize=(12,8))
plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],c = simulated_labels, alpha = .4)

data_x1=np.concatenate((np.ones((num_observations,1)),x1),axis=1)
data_x2=np.concatenate((np.ones((num_observations,1)),x2),axis=1)

#taking a fifth of the data as training data
x1_training=data_x1[:num_observations//5]
x1_test=data_x1[num_observations//5:num_observations]
x2_training=data_x2[:num_observations//5]
x2_test=data_x2[num_observations//5:num_observations]
data_x=np.concatenate((x1_training,x2_training))
labels_training=np.hstack((-np.ones(num_observations//5),np.ones(num_observations//5)))
labels_training_2=np.hstack((-np.zeros(num_observations//5),np.ones(num_observations//5)))

w=private_logistic_regression(0.01,data_x,labels_training,0.1,num_steps = 500000, learning_rate = 5e-2)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(fit_intercept=False, C = 1e15)
clf.fit(data_x, labels_training_2)
print(w)
print(clf.intercept_, clf.coef_)

x=range(-5,5)
w_2=clf.coef_[0]
#plot: 1/2=p(y=1)=w[0]*1+w[1]*x[1]+w[2]*x[2]==>x[2]=-(w[0]-1/2+w[1]*x[1])/w[2]
#first our result
plt.plot(x,-(w[0]-1/2+w[1]*x)/w[2])
#then preimplemented one
plt.plot(x,-(w_2[0]-1/2+w_2[1]*x)/w_2[2],color='red')
plt.show()
