import numpy as np
import matplotlib.pyplot as plt
#todo plot train and test seperately

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
num_epsilons=4
epsilon_array=[100,10,1,0.5]
num_set_sizes=20
error_train=np.zeros((num_epsilons,num_set_sizes))
error_test=np.zeros((num_epsilons,num_set_sizes))

num_observations=num_set_sizes*50
np.random.seed(12)
x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)
data_x1 = np.concatenate((np.ones((num_observations, 1)), x1), axis=1)
data_x2 = np.concatenate((np.ones((num_observations, 1)), x2), axis=1)

simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
# our algorithm assumes the labels are in {-1,1}
simulated_labels = np.hstack((-np.ones(num_observations), np.ones(num_observations)))
# sklearn's algorithm assumes the labels are in {0,1}
simulated_labels_2 = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

for eps in range(num_epsilons):
    for i in range(num_set_sizes):

        num_observations = (i+1)*50
        train_size=num_observations//5
        epsilon=epsilon_array[eps]
        #plt.figure(figsize=(12,8))
        #plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],c = simulated_labels, alpha = .4)


        #taking a fifth of the data as training data
        x1_training=data_x1[:train_size]
        x1_test=data_x1[train_size:num_observations]
        x2_training=data_x2[:train_size]
        x2_test=data_x2[train_size:num_observations]
        data_x=np.concatenate((x1_training,x2_training))
        data_x_test=np.concatenate((x1_test,x2_test))
        labels_training=np.hstack((-np.ones(train_size),np.ones(train_size)))
        labels_training_2=np.hstack((-np.zeros(train_size),np.ones(train_size)))
        labels_test=np.hstack((-np.zeros(num_observations-train_size),np.ones(num_observations-train_size)))
        for j in range(10):
            w=private_logistic_regression(0.01,data_x,labels_training,epsilon,num_steps = 500000, learning_rate = 5e-2)
        #x = range(-5, 5)
        # plot: 1/2=p(y=1)=w[0]*1+w[1]*x[1]+w[2]*x[2]==>x[2]=-(w[0]-1/2+w[1]*x[1])/w[2]
        # first our result

        #plt.plot(x, -(w[0] - 1 / 2 + w[1] * x) / w[2])
        #plt.show()
            y_predicted_train=[int(1/2<=w[0]+w[1]*data_x[i,1]+w[2]*data_x[i,2]) for i in range(train_size*2)]
            y_predicted_test=[int(1/2<=w[0]+w[1]*data_x_test[i,1]+w[2]*data_x_test[i,2]) for i in range((num_observations-train_size)*2)]
            error_train[eps,i]+=sum(abs(y_predicted_train-labels_training_2))/len(y_predicted_train)
            error_test[eps,i]+=sum(abs(y_predicted_test-labels_test))/len(y_predicted_test)
        error_train[eps,i]=error_train[eps,i]/10
        error_test[eps,i]=error_test[eps,i]/10
ind=range(100,(num_set_sizes+1)*100,100)

file = open('error_values.txt','w',encoding='utf8')
file.write('Training error'+'   '+'Test error'+'\n')
for eps in range(num_epsilons):
    for i in range(num_set_sizes):
        file.write(str(error_train[eps,i])+'   '+str(error_test[eps,i])+'\n')
file.close()


num_epsilons=4
epsilon_array=[100,10,1,0.5]
num_set_sizes=20
error_train=np.zeros((num_epsilons,num_set_sizes))
error_test=np.zeros((num_epsilons,num_set_sizes))
ind=range(100,(num_set_sizes+1)*100,100)

"""
file = open('error_values.txt','r',encoding='utf8')
print(file.readline())
for eps in range(num_epsilons):
    for i in range(num_set_sizes):
        line=file.readline()
        s=line.split()
        print(s)
        error_train[eps,i]=s[0]
        error_test[eps,i]=s[1]
"""

fig = plt.figure(figsize=(20,10))
plt.rc('font',size=20)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set(title="Training Error", xlabel="Size of Data Set", ylabel="Misclassification Rate")
ax2.set(title="Test Error", xlabel="Size of Data Set", ylabel="Misclassification Rate")

ax1.plot(ind[5:],error_train[0,5:],color="blue",label="eps=100")
ax1.plot(ind[5:],error_train[1,5:],color="green",label="eps=10")
ax1.plot(ind[5:],error_train[2,5:],color="red",label="eps=1")
ax1.plot(ind[5:],error_train[3,5:],color="xkcd:purple",label="eps=0.5")
#ax1.rc('font', size=30)
ax1.legend()

ax2.plot(ind[5:],error_test[0,5:],color="blue",label="eps=100")
ax2.plot(ind[5:],error_test[1,5:],color="green",label="eps=10")
ax2.plot(ind[5:],error_test[2,5:],color="red",label="eps=1")
ax2.plot(ind[5:],error_test[3,5:],color="xkcd:purple",label="eps=0.5")
#ax2.rc('font', size=30)
ax2.legend()
plt.show()

#print(y_predicted)

"""
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
#then pre-implemented one
plt.plot(x,-(w_2[0]-1/2+w_2[1]*x)/w_2[2],color='red')
plt.show()
"""