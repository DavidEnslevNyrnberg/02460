import os, scipy, matplotlib, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

# plot enabler
doPLOT = 1
figSaver = 0

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

# CREATE SIMULATED DATA
# fix random seed for comparing results
np.random.seed(10)
# set number of test items and number of students
iTest = 20
nStudent = 100

# Simulate student abilities
sdStu = 0.8
meanStu = 1.5
betaTrue = np.random.normal(meanStu, sdStu, nStudent)
# Simulate test difficulty
sdTest = 0.2
meanTest = 0.7
deltaTrue = np.random.normal(meanTest, sdTest, iTest)

# Plot simulation
if doPLOT == 1:
    fig = plt.figure(0, figsize=(14,3))
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
    if figSaver == 1:
        plt.savefig('plots/wDistribution')
    plt.show()

# print(betaTrue); print(deltaTrue)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now for the simulated parameters, define the probabilities of scoring 1, and draw each result from
# from a Bernoulli distribution
pRasch = np.zeros((nStudent, iTest))
SimRaschD = np.zeros((nStudent, iTest))

for n in range (nStudent):
    for i in range(iTest):
        pRasch[n,i] = np.exp(betaTrue[n]-deltaTrue[i])/(1+np.exp(betaTrue[n]-deltaTrue[i]))
        SimRaschD[n,i] = np.random.binomial(1, pRasch[n,i])


# print(pRasch)
dfpRasch = pd.DataFrame(pRasch)
# print(SimRaschD)
dfSimRaschD=pd.DataFrame(SimRaschD)
# dfpRasch.head(5)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set which input Data should be used
inputRasch = dfSimRaschD
print(inputRasch.shape)

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

# Optional: set regularization parameter
lam = 0
# Find parameters to minimize the negative loglikelihood
optimize = minimize(Rasch_Log_Likelihood, w, args=(inputRasch,lam), jac=gradient, options={'maxiter':5000000, 'disp':True})
w_new = optimize.x
w_message = optimize.message
w_success = optimize.success

# print(deltaTrue);
delta_w = w_new[nStudent:]
beta_w = w_new[:nStudent]

if doPLOT==1:
    # plots for real weights against estimated weights
    plt.figure(1, figsize=[6, 6])
    # plt.tight_layout()
    # plt.subplot(2,1,1)
    plt.plot(delta_w, deltaTrue, 'bx', deltaTrue, deltaTrue, 'g-')
    plt.title(r'[i=%d] True $\delta_i$ versus Estimated $\delta_i$' % iTest)
    plt.xlabel(r'True $\delta_i$')
    plt.ylabel(r'Estimated $\delta_i$')
    plt.legend(['est_vs_true', 'trueLine'], loc=2)
    if figSaver == 1:
        plt.savefig('plots/EstimatedDeltaNonPriv')
    plt.show()
    # plt.subplot(2,1,2)
    plt.figure(2, figsize=[6, 6])
    plt.plot(beta_w, betaTrue, 'bx', betaTrue, betaTrue, 'g-')
    plt.title(r'[n=%d] True $\beta_n$ versus Estimated $\beta_n$' % nStudent)
    plt.xlabel(r'True $\beta_n$')
    plt.ylabel(r'Estimated $\beta_n$')
    plt.legend(['est_vs_true', r'true $\beta_n$'], loc=2)
    if figSaver == 1:
        plt.savefig('plots/EstimatedBetaNonPriv')
    plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pRaschEST = np.zeros((nStudent, iTest))
for n in range(nStudent):
    for i in range(iTest):
        pRaschEST[n,i] = np.exp(beta_w[n]-delta_w[i])/(1+np.exp(beta_w[n]-delta_w[i]))

if doPLOT==1:
    # plots for real weights against estimated weights
    plt.figure(4, figsize=[10, 10])
    plt.tight_layout()
    plt.plot(pRasch[:100], pRaschEST[:100], 'bx', markersize=8)
    plt.plot(pRasch[1:], pRasch[1:], 'g-', linewidth=7)
    plt.title(r'[N=%d]  $p(X_{ni}=1)_{est}$ vs $p(X_{ni}=1)_{true}$' % nStudent, fontdict={'fontsize': 30})
    plt.xlabel(r'$p(X_{ni}=1)_{true}$', fontdict={'fontsize': 30})
    plt.ylabel(r'$p(X_{ni}=1)_{est}$', fontdict={'fontsize': 30})
    plt.legend(['Estimated', 'True'], loc=2, fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    if figSaver == 1:
        plt.savefig('plots/EstimatedVsTrueNonPriv')
    plt.show()

# plt.scatter(pRasch[:100], pRaschEST[:100])
# plt.plot(pRasch[:100], pRasch[:100])
# plt.show()
