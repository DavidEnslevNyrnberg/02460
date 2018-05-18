import os, scipy, matplotlib, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

# plot enabler
doPLOT = 0

# Loading of real data
Dir = os.getcwd()
gradeDir = Dir+r'\Data\final_grades.xlsx'
intermediateDir = Dir+r'\Data\intermediate_grades.xlsx'

dfGrade = pd.read_excel(gradeDir, sheet_name=r'Exam (Second time)')
dfGrade.head()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Converting to input form of Rasch model
headQ = list(dfGrade)[1:]
normPoint = dfGrade[headQ].max()[0:16]
bolGrade = (dfGrade[headQ]/normPoint).round()

dfData = dfGrade.copy()
dfData[headQ] = bolGrade[headQ]
dfRaschD = dfData[headQ[0:-1]]
# dfRaschD.head(5)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# print(betaTrue); print(deltaTrue)

# ~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set which input Data should be used
inputRasch = dfRaschD
(nStudent,iTest)=inputRasch.shape

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set initial gradient guess to zero
w_initial = np.zeros(nStudent + iTest)
w = w_initial

# A function to return the negative Loglikelihood of the Rasch model to be minimized
def Rasch_Log_Likelihood(w, data_y,lam):
    (N, I) = data_y.shape
    wBeta = w[0:N]
    wDelta = w[N:N + I]

    wMatrix = np.array([[n - i for n in wBeta] for i in wDelta])
    wExpMatrix = np.exp(wMatrix)
    wlogMatrix = np.log(1+wExpMatrix)
    likelihood = -np.sum(np.sum(wlogMatrix,axis=1), axis=0)+np.sum(np.multiply(wDelta, np.sum(data_y, axis=0)))-np.sum(np.multiply(wBeta, np.sum(data_y, axis=1)))+lam*w.T.dot(w)
    return(likelihood)

# a function to return the gradient of the negative Loglikelihood; used for minimizing
def gradient(w, data_y, lam):
    (N, I) = data_y.shape
    wBeta = w[0:N].reshape([N, 1])  # w_beta
    wDelta = w[N:N + I].reshape([1, I])  # w_delta

    wMatrix = np.array([n - i for n in wBeta for i in wDelta])
    wExpMatrix = scipy.special.expit(wMatrix)

    gradBeta = np.sum(wExpMatrix, axis=1) - np.sum(data_y, axis=1)+2*lam*w[0:N]
    gradDelta = -np.sum(wExpMatrix, axis=0) + np.sum(data_y, axis=0)+2*lam*w[N:N+I]

    w_gradient = np.concatenate([gradBeta, gradDelta])
    return w_gradient


def Private_Rasch_Log_Likelihood(w, data_y,lam,b):
    (N, I) = data_y.shape
    wBeta = w[0:N]#.reshape([N, 1])  # w_beta
    wDelta = w[N:N + I]#.reshape([1, I])  # w_delta

    wMatrix = np.array([[n - i for n in wBeta] for i in wDelta])
    wExpMatrix = np.exp(wMatrix)
    wlogMatrix = np.log(1+wExpMatrix)
    likelihood=-np.sum(np.sum(wlogMatrix,axis=1),axis=0)+np.sum(np.multiply(wDelta,np.sum(data_y, axis=0)))-np.sum(np.multiply(wBeta,np.sum(data_y,axis=1)))+lam*w.T.dot(w)+b.T.dot(wDelta)
    return(likelihood)

# a function to return the gradient of the negative Loglikelihood; used for minimizing
def Private_gradient(w, data_y,lam,b):
    (N, I) = data_y.shape
    wBeta = w[0:N].reshape([N, 1])  # w_beta
    wDelta = w[N:N + I].reshape([1, I])  # w_delta

    wMatrix = np.array([n - i for n in wBeta for i in wDelta])
    wExpMatrix = scipy.special.expit(wMatrix)

    gradBeta = np.sum(wExpMatrix, axis=1) - np.sum(data_y, axis=1)+2*lam*w[0:N]
    gradDelta = -np.sum(wExpMatrix, axis=0) + np.sum(data_y, axis=0)+2*lam*w[N:N+I]+b

    w_gradient = np.concatenate([gradBeta, gradDelta])
    return w_gradient

# Optional: set regularization parameter
lam = 0

pRaschEST = np.zeros((4, iTest, nStudent))
# Find parameters to minimize the negative loglikelihood
optimize = minimize(Rasch_Log_Likelihood, w, args=(inputRasch,lam), jac=gradient, options={'maxiter':5000000, 'disp':True})
w_new = optimize.x
w_message = optimize.message
w_success = optimize.success

# print(deltaTrue);
delta_w = w_new[nStudent:]
beta_w = w_new[:nStudent]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for n in range(nStudent):
    for i in range(iTest):
        pRaschEST[0,i,n] = np.exp(beta_w[n]-delta_w[i])/(1+np.exp(beta_w[n]-delta_w[i]))

for j in range(3):
    epsilon=10^(j+1)
    b_norm = np.random.gamma(iTest, scale=np.sqrt(iTest) / epsilon)
    print("b_norm" + str(b_norm))
    # b_norm=0
    # then direction randomly in d-dimensional space (http://mathworld.wolfram.com/HyperspherePointPicking.html)
    bx = np.random.normal(size=iTest)
    b = bx / np.linalg.norm(bx) * b_norm
    # Find parameters to minimize the negative loglikelihood
    optimize = minimize(Private_Rasch_Log_Likelihood, w, args=(inputRasch, lam, b), jac=Private_gradient,
                        options={'maxiter': 5000000, 'disp': True})
    w_new = optimize.x
    w_message = optimize.message
    w_success = optimize.success

    # print(deltaTrue);
    delta_w = w_new[nStudent:]
    beta_w = w_new[:nStudent]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for n in range(nStudent):
        for i in range(iTest):
            pRaschEST[j+1,i, n] = np.exp(beta_w[n] - delta_w[i]) / (1 + np.exp(beta_w[n] - delta_w[i]))

#x=range(1,iTest+1)
#x=np.linspace(0,1,20)
plt.figure(2, figsize=[18, 8])
plt.tight_layout()
plt.scatter(pRaschEST[0,:,:],pRaschEST[1,:,:], c='g', marker='x',linewidths=5)
plt.scatter(pRaschEST[0,:,:],pRaschEST[2,:,:], c='b', marker='x',linewidths=5)
plt.scatter(pRaschEST[0,:,:],pRaschEST[3,:,:], c='r', marker='x',linewidths=5)
x=np.linspace(0,1,20)
plt.plot(x,x,'k-')
plt.title(r'Estimated Probabilities - Data size: [N=%d, I=%d]' % (nStudent, iTest), fontdict={'fontsize': 24})
plt.xlabel(r'Non-Private Estimates', fontdict={'fontsize': 20})
plt.ylabel(r'Private Estimates', fontdict={'fontsize': 20})
plt.legend(['correlation=1','eps=10', 'eps=100','eps=1000'], loc=3, fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.savefig('StudentExample')
plt.show()