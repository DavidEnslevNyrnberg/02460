import os, scipy, matplotlib, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

Dir = os.getcwd()
gradeDir = Dir+r'\Data\final_grades.xlsx'
intermediateDir = Dir+r'\Data\intermediate_grades.xlsx'

dfGrade = pd.read_excel(gradeDir, sheet_name=r'Exam (Second time)')
dfGrade.head()

# ~~~~~~~~~~~~~~~~~~~~

headQ = list(dfGrade)[1:]
# print(headQ)

normPoint = dfGrade[headQ].max()[0:16]
# print(normPoint)

bolGrade = (dfGrade[headQ]/normPoint).round()
# print(bolGrade)

dfData = dfGrade.copy()
dfData[headQ] = bolGrade[headQ]
dfRaschD = dfData[headQ[0:-1]]
dfRaschD.head(5)

# ~~~~~~~~~~~~~~~~~~~~

# import string

np.random.seed(10)

iTest = 100
nStudent = 1000
# letters = list(string.ascii_lowercase[:jTest])

# sim students
sdStu = 0.8
meanStu = 1.5
betaTrue = np.random.normal(meanStu, sdStu, nStudent)
# sim test difficulty
sdTest = 0.2
meanTest = 0.7
deltaTrue = np.random.normal(meanTest, sdTest, iTest)

#plot simmulation
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

outputRasch = dfSimRaschD
print(outputRasch.shape)

# ~~~~~~~~~~~~~~~~~~~~

w_initial = np.zeros(nStudent + iTest)
w = w_initial
"""
step_size = 0.05
w_gradient = np.ones(nStudent + iTest)

gradBeta = np.ones(nStudent)
gradDelta = np.ones(iTest)
"""

def Rasch_Log_Likelihood(w, data_y,lam):
    (N, I) = data_y.shape
    wBeta = w[0:N]#.reshape([N, 1])  # w_beta
    wDelta = w[N:N + I]#.reshape([1, I])  # w_delta

    wMatrix = np.array([[n - i for n in wBeta] for i in wDelta])
    wExpMatrix = np.exp(wMatrix)
    wlogMatrix = np.log(1+wExpMatrix)
    likelihood=-np.sum(np.sum(wlogMatrix,axis=1),axis=0)+np.sum(np.multiply(wDelta,np.sum(data_y, axis=0)))-np.sum(np.multiply(wBeta,np.sum(data_y,axis=1)))+lam*w.T.dot(w)
    return(likelihood)

def gradient(w, data_y,lam):
    (N, I) = data_y.shape
    wBeta = w[0:N].reshape([N, 1])  # w_beta
    wDelta = w[N:N + I].reshape([1, I])  # w_delta

    wMatrix = np.array([n - i for n in wBeta for i in wDelta])
    wExpMatrix = scipy.special.expit(wMatrix)

    gradBeta = np.sum(wExpMatrix, axis=1) - np.sum(data_y, axis=1)+2*lam*w[0:N]
    gradDelta = -np.sum(wExpMatrix, axis=0) + np.sum(data_y, axis=0)+2*lam*w[N:N+I]

    w_gradient = np.concatenate([gradBeta, gradDelta])
    return w_gradient

data_y=outputRasch
(N,I)=data_y.shape
#L=Rasch_Log_Likelihood(w_gradient, data_y)
#print(L)
lam=0
optimize=minimize(Rasch_Log_Likelihood,w,args=(outputRasch,lam),jac=gradient, options={'maxiter': 5000000, 'disp': True})
w_new=optimize.x
w_message=optimize.message
w_success=optimize.success


print(deltaTrue);
delta_w=w_new[nStudent:]
beta_w=w_new[:nStudent]

#plt.scatter(delta_w,deltaTrue)
#plt.show()

plt.scatter(beta_w, betaTrue)
plt.show()
# optimizer for gradient decent - Martin link to optimizer
"""
step = 0
while (np.linalg.norm(w_gradient)>10**(-10)) & (step<500000):
        # Update weights with gradient descent
        w_gradient = gradient(data_y=outputRasch, w=w)
        w -= step_size*w_gradient
        if step%1000==0:
            print(step)
        step+=1

wTrue = np.concatenate((betaTrue, deltaTrue))
print(wTrue[nStudent:]); print(w[nStudent:])
"""
# ~~~~~~~~~~~~~~~~~~~~

#print('finish')