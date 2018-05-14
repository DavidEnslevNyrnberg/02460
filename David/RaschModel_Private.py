import os, scipy, matplotlib, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
DOPLOTS=
#Loading of real data
Dir = os.getcwd()
gradeDir = Dir+r'\Data\final_grades.xlsx'
intermediateDir = Dir+r'\Data\intermediate_grades.xlsx'

dfGrade = pd.read_excel(gradeDir, sheet_name=r'Exam (Second time)')
dfGrade.head()

# ~~~~~~~~~~~~~~~~~~~~
#converting to input form of Rasch model
headQ = list(dfGrade)[1:]
normPoint = dfGrade[headQ].max()[0:16]
bolGrade = (dfGrade[headQ]/normPoint).round()

dfData = dfGrade.copy()
dfData[headQ] = bolGrade[headQ]
dfRaschD = dfData[headQ[0:-1]]
#dfRaschD.head(5)

# ~~~~~~~~~~~~~~~~~~~~

#CREATE SIMULATED DATA
#fix random seed for comparing results
np.random.seed(10)
#set number of test items and number of students
iTest = 16
nStudent = 62

# simulate student abilities
sdStu = 0.8
meanStu = 1.5
betaTrue = np.random.normal(meanStu, sdStu, nStudent)
# simulate test difficulty
sdTest = 0.2
meanTest = 0.7
deltaTrue = np.random.normal(meanTest, sdTest, iTest)

#plot simulation
"""
fig = plt.figure(figsize=(14,3))
fig.tight_layout()
ax1 = fig.add_subplot(131)
ax1.set_xlabel(r'Ability $\beta_n$')
ax1.set_ylabel('Percent of student')
ax1.set_title("$n=100$, Student ability distribution")
ountStu, binsStu, ignoredStu = plt.hist(betaTrue, bins=20, normed=True)
plt.plot(binsStu, 1/(sdStu*np.sqrt(2*np.pi))*np.exp(-(binsStu-meanStu)**2/(2*sdStu**2)), linewidth=2, color='r')
ax2 = fig.add_subplot(132)
ax2.set_xlabel(r'Difficulty, $\delta_i$')
ax2.set_ylabel('Percent of questions')
ax2.set_title("$i=20$, Test difficulty distribution")
ountTest, binsTest, ignoredTest = plt.hist(deltaTrue, bins=20, normed=True)
plt.plot(binsTest, 1/(sdTest*np.sqrt(2*np.pi))*np.exp(-(binsTest-meanTest)**2/(2*sdTest**2)), linewidth=2, color='r')
"""
# plt.show()
# print(betaTrue); print(deltaTrue)

# ~~~~~~~~~~~~~~~~~~~~

#now for the simulated parameters, define the probabilities of scoring 1, and draw each result from
#from a Bernoulli distribution
pRasch = np.zeros((nStudent,iTest))
SimRaschD = np.zeros((nStudent,iTest))

for n in range (nStudent):
    for i in range(iTest):
        pRasch[n,i] = np.exp(betaTrue[n]-deltaTrue[i])/(1+np.exp(betaTrue[n]-deltaTrue[i]))
        SimRaschD[n,i] = np.random.binomial(1,pRasch[n,i])


#print(pRasch)
dfpRasch = pd.DataFrame(pRasch)
#print(SimRaschD)
dfSimRaschD=pd.DataFrame(SimRaschD)
#dfpRasch.head(5)

# ~~~~~~~~~~~~~~~~~~~~
#set which input Data should be used
inputRasch = dfRaschD
print(inputRasch.shape)

# ~~~~~~~~~~~~~~~~~~~~
#set initial gradient guess to zero
w_initial = np.zeros(nStudent + iTest)
w = w_initial

# a function to return the negative Loglikelihood of the Rasch model to be minimized
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

#optional: set regularization parameter
lam=0
#define noise vector
# first, draw a noise vector b distributed like exp(-eps/2*||b||)
# to do this, first pick the norm according to gamma distribution:
epsilon=10
b_norm = np.random.gamma(iTest, scale=iTest / epsilon)
print("b_norm" + str(b_norm))
# b_norm=0
# then direction randomly in d-dimensional space (http://mathworld.wolfram.com/HyperspherePointPicking.html)
bx = np.random.normal(size=iTest)
b = bx / np.linalg.norm(bx) * b_norm
#find parameters to minimize the negative loglikelihood
optimize=minimize(Private_Rasch_Log_Likelihood,w,args=(inputRasch,lam,b),jac=Private_gradient, options={'maxiter': 5000000, 'disp': True})
w_new=optimize.x
w_message=optimize.message
w_success=optimize.success

#print(deltaTrue);
delta_w=w_new[nStudent:]
beta_w=w_new[:nStudent]

pRaschEST = np.zeros((nStudent,iTest))
for n in range (nStudent):
    for i in range(iTest):
        pRaschEST[n,i] = np.exp(beta_w[n]-delta_w[i])/(1+np.exp(beta_w[n]-delta_w[i]))

file = open('probabilities.txt','w',encoding='utf8')
n=10
for i in range(iTest):
    file.write(str(pRaschEST[n,i])+' & ')
file.write('\n')
"""
plt.scatter(pRasch[:100],pRaschEST[:100])
plt.plot(pRasch[:100],pRasch[:100])
"""
#plot real delta against estimated delta
#plt.scatter(delta_w,deltaTrue)
#plt.show()

#plot real beta against estimated beta
#plt.scatter(beta_w, betaTrue)
plt.show()