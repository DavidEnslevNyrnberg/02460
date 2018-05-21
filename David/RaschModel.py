import os, scipy, matplotlib, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

# enablers for plots and saving figures
doPLOT = 1
figSaver = 1
# define fixed values
count = 0 # location of distribution settings
rep = 10
iTest = 20
nStudent = 100
lam = 0.01 # regularization parameter
epsilon = 100 # privatization parameter
meanStu = np.linspace(0.4, 0.6, 3)
sdStu = 1
meanTest = 0.5
sdTest = np.linspace(0.8, 1.2, 3)

# initialize result array
pTruepEstCorr = np.zeros([rep,len(meanStu)*len(sdTest)])
pTruepEstCorr[pTruepEstCorr==0]=np.nan
# run code for different distribution settings
for muStu in meanStu:
    for sigTest in sdTest:
        betaTrue = np.random.normal(muStu, sdStu, [rep, nStudent])
        deltaTrue = np.random.normal(meanTest, sigTest, [rep, iTest])
        print('sigTest: %2f \nmuStu: %2f \ncount: %d' % (sigTest, muStu, count))

        # run repetitions to average out results
        for ite in range(rep):
            pRasch = np.zeros((nStudent, iTest))
            SimRaschD = np.zeros((nStudent, iTest))
            # calculate Rasch model and Rasch data
            for n in range(nStudent):
                for i in range(iTest):
                    pRasch[n, i] = np.exp(betaTrue[ite, n] - deltaTrue[ite, i]) / (1 + np.exp(betaTrue[ite, n] - deltaTrue[ite, i]))
                    SimRaschD[n, i] = np.random.binomial(1, pRasch[n, i])
            # change to pd.dataframe
            dfpRasch = pd.DataFrame(pRasch)
            dfSimRaschD = pd.DataFrame(SimRaschD)

            inputRasch = dfSimRaschD
            # initial weight vector
            w_initial = np.random.normal(0, 1, [nStudent + iTest])*0.001
            w = w_initial.copy()

            # optimize the Rasch negativ log likelihood by the gradient
            optimize = minimize(Rasch_Log_Likelihood, w, args=(inputRasch,lam), jac=gradient, options={'maxiter':5000000, 'disp':False})

            # omit optimized step with no iteration
            if optimize.nit == 0:
                continue
            w_new = optimize.x
            w_message = optimize.message
            w_success = optimize.success

            print(r'Included optimzed at rep: %d' % ite)

            delta_w = w_new[nStudent:]
            beta_w = w_new[:nStudent]

            # initialize estimated rasch model.
            pRaschEST = np.zeros((nStudent, iTest))
            for n in range(nStudent):
                for i in range(iTest):
                    pRaschEST[n, i] = np.exp(beta_w[n] - delta_w[i]) / (1 + np.exp(beta_w[n] - delta_w[i]))

            vec_pRasch = pRasch.reshape([1, nStudent*iTest]).copy()
            vec_pRaschEST = pRaschEST.reshape([1,nStudent*iTest]).copy()
            Cov_nonPriv = np.cov(vec_pRasch,vec_pRaschEST)
            pTruepEstCorr_nonPriv[ite,count] = Cov[0][1]/np.sqrt(Cov[0][0]*Cov[1][1])

            if doPLOT == 1 and count in [1,3,4,5,7]:
                # plots for real weights against estimated weights
                plt.figure(ite+10, figsize=[10, 10])
                plt.tight_layout()
                plt.scatter(vec_pRasch, vec_pRaschEST, c='b', marker='x', s=40)
                x = np.linspace(0,1,20)
                plt.plot(x,x,'g-', linewidth=7)
                plt.title(r'$\mu_{student}= %.2f \qquad \sigma_{test}= %.2f$' % (muStu,sigTest), fontdict={'fontsize': 30})
                plt.xlabel(r'$p(X_{ni}=1)_{true}$', fontdict={'fontsize': 30})
                plt.ylabel(r'$p(X_{ni}=1)_{est}$', fontdict={'fontsize': 30})
                plt.legend(['correlation=1','Estimated vs. True'], loc=2, fontsize=20)
                plt.tick_params(axis='both', labelsize=20)
                if figSaver == 1:
                    plt.savefig(r'valPlot/EstimatedVsTrueNonPriv_ite%d'%(ite+10*count))
                    print(r'repNr: %d' % ite)
                    print(r'parameter: %d' % count)
                plt.show()
        count += 1

CorrMatrix_NonPriv = np.nanmean(pTruepEstCorr, axis=1)

# fee = [np.mean(y) for y in foo if not y==0]
print('test code')
