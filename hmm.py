"""
homework 3: PGM
@author: eric.benhamou
"""

import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.stats import multivariate_normal


COLORS = ["r", "orange", "b", "g", "darkgreen" ]
XLIM = YLIM = np.array([-10,10])    # Graph limits
DPI = 300                           # Image resolution
STEPS = 1000                        # Step of conics plotting
ALPHA = .1                          # 90% confidence interval

#%% Generic function to plot an ellipse
def plot_ellipse(mu, Q, c, color="k", label=""):
    """Plots conic of equation (x-mu).T * Q * (x-mu) = c. mu and Q
    are expected to be matrices."""
    X = np.linspace(XLIM[0], XLIM[1], STEPS)
    Y = np.linspace(XLIM[0], XLIM[1], STEPS)
    X, Y = np.meshgrid(X, Y) # X and Y are ARRAYS
    X_ = X - mu[0,0]
    Y_ = Y - mu[1,0]
    res = X_*X_*Q[0,0] + 2*X_*Y_*Q[0,1] + Y_*Y_*Q[1,1] - c
    plt.contour(X, Y, res, [0], colors=color, linewidths=3)
    plt.plot([XLIM[0]-1], [YLIM[0]-1], color,
             linestyle="-",lw=3, label=label) #Fake label
    plt.xlim(XLIM)
    plt.axis('equal')
    return

"""
Hidden Markov Model class
assuming Normal emission probabilities
"""
class Hmm(object):
    def __init__(self, hmm_type = 'log-scale' ):

        if hmm_type != 'log-scale' and hmm_type != 'rescaled':
            raise RuntimeError("unknow type! allowed log-scale or rescaled")

        # the initial probability distribution
        self.K = 4
        self.states = range(self.K)
        self.pi = np.full((self.K), 1 / self.K)
        # the probability transition matrix
        # stays 0.5 in its current state and 0.5/(self.K-1) otherwise
        self.a = np.matrix([[0.5 if i == j else 0.5 / (self.K - 1)
                             for j in range(self.K)] for i in range(self.K)])

        # the emission probabilities are supposed to be Gaussian 
        # with parameters  Normal(\Mu_i,\Sigma_i)
        self.mu = np.array([
            [-3.06194232, -3.5345231],
            [3.800698780, -3.7971300],
            [-2.03436423, 4.17258868],
            [3.977956380, 3.77384145]
            ])
        self.sigma = np.array([
            [[6.24150219, 6.05026172],
             [6.05026172, 6.18253658]],
            [[0.92120586, 0.05736506],
             [0.05736506, 1.86620842]],
            [[2.90443607, 0.20656898],
             [0.20656898, 2.75617571]],
            [[0.21034075, 0.29033321],
             [0.29033321, 12.23765529]]
            ])
        self.pi_mixture = np.array([0.2474, 0.0355, 0.4523, 0.2649])
        #store the initial value for question 6 where we 
        #compute the log likelihood for mixture
        self.a0 = self.a
        self.pi0 = self.pi
        self.mu0 = self.mu
        self.sigma0 = self.sigma 
        
        self.train_data, self.T = self.__read_data('EMGaussian.data')
        self.test_data, dummy = self.__read_data('EMGaussian.test')
        self.hmm_type=hmm_type
        return

    # function to read the data and return data and size
    def __read_data(self, filename):
        data = pd.read_csv(filename, delimiter=' ',
                                header=None).values[:, :]
        T = len(data)
        return data, T

    # compute the emissio probabilities
    # based on gaussian assumptions
    def __compute_B(self, data ):
        self.normals = [multivariate_normal(
            mean=self.mu[i, :], cov=self.sigma[i, :, :]) for i in range(self.K)]
        T = len(data)
        self.b = np.zeros((T, self.K))
        for t in range(T):
            self.b[t, :] = [self.normals[y].pdf(
                data[t, :]) for y in range(self.K)]
        
        #other computation for log-scale
        if self.hmm_type == 'log-scale':
            self.log_b = np.zeros((T, self.K))
            for t in range(T):
                self.log_b[t, :] = np.log(self.b[t, :])
        return

    # Q1: alpha recursion (this is a forward recursion)
    # implemented either in log-scale or rescaled version
    def __alpha_recursion(self):
        if self.hmm_type == 'log-scale':
            self.log_alpha = np.zeros((self.T, self.K))
            # Initialize base cases (t == 0)
            self.log_alpha[0, :] = np.log(self.pi * self.b[0, :])

            # Run Forward algorithm for t > 0
            for t in range(self.T-1):
                log_alpha_star = np.amax(self.log_alpha[t, :])
                for q in self.states:
                    self.log_alpha[t + 1, q] = self.log_b[t + 1, q] + log_alpha_star + \
                        math.log(sum((math.exp(self.log_alpha[t, q0] -
                                               log_alpha_star) * self.a[q0, q])for q0 in self.states))
        # rescaled case
        elif self.hmm_type == 'rescaled':
            self.alpha = np.zeros((self.T, self.K))
    
            # Initialize base cases (t == 0)
            self.alpha[0, :] = self.pi * self.b[0, :] 
            s = self.alpha[0,:].sum()
            self.alpha[0,:] = self.alpha[0,:]/s
            
            # Run Forward algorithm for t > 0
            for t in range(self.T-1):
                self.alpha[t+1,:] = np.multiply(self.b[t+1,:], np.dot(self.a.T, self.alpha[t,:]))
                s = self.alpha[t+1,:].sum()
                self.alpha[t+1,:] = self.alpha[t+1,:]/s
        else:
            raise RuntimeError("unknow type allowed log-scale or rescaled")
        return

    # beta recursion (this is a backward recursion)
    # implemented either in log-scale or rescaled version
    def __beta_recursion(self):
        if self.hmm_type == 'log-scale':
            self.log_beta = np.zeros((self.T, self.K))
            for q in self.states:
                self.log_beta[self.T - 1, q] = 0
            for t in reversed(range(self.T - 1)):
                log_beta_star = np.amax(self.log_beta[t + 1, :])
                for q in self.states:
                    self.log_beta[t, q] = log_beta_star + math.log(sum((math.exp(
                        self.log_beta[t + 1, q1] - log_beta_star) * self.a[q, q1] * self.b[t + 1, q1]) for q1 in self.states))
        elif self.hmm_type == 'rescaled':
            self.beta = np.zeros((self.T, self.K))
            self.scale_factor_beta = np.ones(self.T)
            self.beta[self.T - 1,:] = 1/self.K
            self.scale_factor_beta[self.T - 1] = self.K
            
            for t in range(self.T-2, -1, -1):
                self.beta[t,:] = np.dot(self.a, np.multiply(self.b[t+1, :], self.beta[t+1, :]))
                self.scale_factor_beta[t] = self.beta[t,:].sum()
                self.beta[t,:] = self.beta[t,:] / self.scale_factor_beta[t] 
        else:
            raise RuntimeError("unknow type allowed log-scale or rescaled")

        return

    # Q2: compute the conditional probabitilies
    # based on the data
    # called the function to compute the emission probabilities
    # as well as the alpha and beta recursion
    # implemented either in log-scale or rescaled version
    def compute_proba(self, data = np.array([])):
        if data.size == 0:
            data = self.train_data
            
        self.__compute_B(data)
        self.__alpha_recursion()
        self.__beta_recursion()

        #initialize array
        self.cond_proba = np.zeros((self.T, self.K))
        self.joined_cond_proba = np.zeros((self.T-1, self.K, self.K))
        
        # do the computation
        if self.hmm_type == 'log-scale':
            for t in range(self.T):
                amax = np.max(self.log_alpha[t, :] + self.log_beta[t, :])
                proba_sum = sum((math.exp(
                    self.log_alpha[t, zt] + self.log_beta[t, zt] - amax)) for zt in self.states)
                for q in self.states:
                    self.cond_proba[t, q] = math.exp(
                        self.log_alpha[t, q] + self.log_beta[t, q] - amax) / proba_sum
                if t  < self.T -1:
                   for q in self.states:
                       for q1 in self.states:
                            self.joined_cond_proba[t, q, q1] = math.exp(
                                self.log_alpha[t, q] + self.log_beta[t + 1, q1] - amax) \
                                    * self.b[t + 1, q1] * self.a[q, q1] / proba_sum
        elif self.hmm_type == 'rescaled':
            self.cond_proba = self.alpha * self.beta
            denom = self.cond_proba.sum(1)
            self.cond_proba = self.cond_proba / denom[:, None]
            
            denom = denom * self.scale_factor_beta
            denom = denom[:(self.T -1), None, None]
            self.joined_cond_proba = self.alpha[:(self.T -1), :, None] * self.beta[1:self.T, None, :] \
                * np.asarray(self.a)[None, :, :] \
                * self.b[1:self.T, :, None].transpose((0, 2, 1)) \
                / denom
        else:
            raise RuntimeError("unknow type allowed log-scale or rescaled")
        return

    #Q2: plot the conditional proba p(q_t|u)    
    def plot_proba(self, T_max, title, prefix, suffix):
        self.__plot_states( self.cond_proba, T_max, title, prefix, suffix)
        return
    
    def plot_most_likely_state(self, path_data, T_max, title, prefix, suffix):
        data = np.zeros((len(path_data),self.K))
        for i in range(len(path_data)):
            data[i, int(path_data[i])] = 1
        self.__plot_states( data, T_max, title, prefix, suffix, 'step')
        return
    
    def __plot_states(self, data, T_max, title, prefix, suffix, plot_type='plot'):
        f, axarr = plt.subplots(self.K,  sharex=True)
        for i in range(self.K):
            if plot_type == 'step':
                axarr[i].step(range(T_max), data[:T_max, i],c=COLORS[i], label="State %d" % (i+1) )
            else:
                axarr[i].plot(range(T_max), data[:T_max, i],c=COLORS[i], label="State %d" % (i+1) )
            axarr[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        f.subplots_adjust(hspace=0.2)
        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title(title, y = (self.K + 0.8))
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        utils.save_figure(plt, prefix, suffix, lgd )
        return

    # the incomplete lok likelihood is composed 
    # of 3 terms as explained in the pdf file
    def expected_complete_log_likelihood(self,data):
        term_1 = (self.cond_proba[0,:] * np.log(self.pi)).sum()
        term_2 = (self.joined_cond_proba * np.log(self.a[None,:,:])).sum()
        term_3 = (self.cond_proba * np.log(self.b)).sum()
        return term_1 + term_2 + term_3


    # Q3 Expectation Maximization algo
    def EM(self, compute_likelihood  = False, max_iter = 200, precision = 1e-4 ):
        if compute_likelihood:
            self.llh = []
           
        for i in range(max_iter):
            pi_0, a_0, mu_0, sigma_0 = self.pi, self.a, self.mu, self.sigma
            # expectation
            self.compute_proba(self.train_data)

            # maximimization
            self.pi = self.cond_proba[0,:]
            self.a = self.joined_cond_proba.sum(0)/ self.joined_cond_proba.sum((0,1))
            mean_cond_proba = self.cond_proba.sum(0)
            self.mu = (self.cond_proba[:,:,None]*self.train_data[:,None,:]).sum(0) / mean_cond_proba[:,None]
            self.sigma = np.zeros((self.K, 2, 2))
            for t in range(self.T):
                self.sigma = self.sigma + self.cond_proba[t,:,None,None] * \
                    (self.train_data[t,None,:]-self.mu)[:,:,None]*(self.train_data[t,None,:]-self.mu)[:,:,None].transpose(0,2,1)
            self.sigma = self.sigma / mean_cond_proba[:,None,None]
            
            #compute expected_complete_log_likelihood
            if compute_likelihood:
                train_llh = self.expected_complete_log_likelihood(self.train_data)
                self.compute_proba(self.test_data)
                test_llh = self.expected_complete_log_likelihood(self.test_data)
                self.llh.append([train_llh, test_llh])
               
            #Check halt condition
            if max(np.max(np.abs(self.pi - pi_0)),
                   np.max(np.abs(self.a - a_0)),
                   np.max(np.abs(self.mu - mu_0)),
                   np.max(np.abs(self.sigma - sigma_0))) < precision:
                break
            if i == max_iter:
                raise RuntimeError("max iteration reached")
        self.iterations = i+1
        return

    # a simple helper function to print the parameters
    def print_parameters(self,precision=4):
        np.set_printoptions(precision)
        print('EM converged in ', self.iterations, ' iterations')
        print('pi:' )
        print(self.pi)
        print('a:')
        print(self.a)
        print('mu:')
        print(self.mu)
        print('sigma:')
        print(self.sigma)
        print('pi difference:')
        print(self.pi-self.pi0)
        print('a difference:')
        print(self.a-self.a0)
        print('mu difference:')
        print(self.mu-self.mu0)
        print('sigma difference:')
        print(self.sigma-self.sigma0)
        
    # Q5 plot expected_complete_log_likelihood
    def plot_likelihood(self, prefix, suffix):
        plt.figure()
        self.llh = np.asarray(self.llh)
        N=len(self.llh)
        labels = ['train llh', 'test llh']
        for i in range(2):
            plt.plot(range(1,(N+1)), self.llh[:,i], c=COLORS[i], label=labels[i])
        plt.ylabel('log likelihood')
        plt.xlabel('EM iterations')
        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        utils.save_figure(plt, prefix, suffix, lgd )
        return

    #Q6 compute the log likelihood for the mixture model
    def __llh_mixture(self, data):
        normals = [multivariate_normal(mean=self.mu0[i, :], cov=self.sigma0[i, :, :]) for i in range(self.K)]
        T=len(data)
        pdfs = np.zeros((T, self.K))
        for t in range(T):
            pdfs[t,:] = [ normals[y].pdf(data[t, :]) for y in range(self.K)]
        return np.log((self.pi_mixture[None,:] * pdfs).sum(axis=1)).sum()

    #Q6: print in latex format the tables for the 
    #two models
    def print_loglikelihoods_table(self):
        #first table
        print('\\begin{table}[H]')
        print('\\begin{center}')
        print('\\begin{tabular}{|r|r|r|}')
        print('\hline')
        print('\diagbox[width=8em]{Iterations}{Dataset} & HMM Train & HMM Test \\\\ \hline')
        N=len(self.llh)
        for i in range(N):
            print( i+1, ' & ', '{:.1f}'.format(self.llh[i, 0]), ' & ',  '{:.1f}'.format(self.llh[i, 1]), '\\\\ \hline')
        print('\end{tabular}')
        print('\caption{likelihood table for the HMM model in terms of EM iterations}')
        print('\label{table:q5}')
        print('\end{center}')
        print('\end{table}')
        
        #second table
        print('\n\n')
        print('\\begin{table}[H]')
        print('\\begin{center}')
        print('\\begin{tabular}{|r|r|r|}')
        print('\hline')
        print('\diagbox[width=8em]{Model}{Dataset} & Train & Test \\\\ \hline')
        print('HMM & ', '{:.0f}'.format(self.llh[N-1, 0]), ' & ',  '{:.0f}'.format(self.llh[N-1, 1]), '\\\\ \hline')
        print('Mixture  & ', '{:.0f}'.format(self.__llh_mixture(self.train_data)), ' & ',  '{:.0f}'.format(self.__llh_mixture(self.test_data)), '\\\\ \hline')
        print('\end{tabular}')
        print('\caption{likelihood table}')
        print('\label{table:q6}')
        print('\end{center}')
        print('\end{table}')
        return
    

    #Q7    
    # Viterbi decoding algorithm to find the most probable path followed
    # by the latent variables Q to generate the observations u
    def compute_viterbi_path(self, data):
        T=len(data)
        self.path = np.zeros(T)
        self.max_index = np.zeros((T, self.K))
        self.max_proba = np.zeros((T, self.K))

        if self.hmm_type == 'log-scale':
            # precompute the log
            l_pi = np.log(self.pi)
            l_a = np.log(self.a)
            l_b = np.log(self.b)
            #in log scale the max proba is the log of the max proba
            self.max_proba[0,:] = l_pi + l_b[0,:]
            
            # Run Viterbi for t > 0
            for t in range(1, T):
                for q in self.states:
                    (self.max_proba[t, q], self.max_index[t-1, q]) = max((self.max_proba[t-1, q0] + l_a[q0, q] + l_b[t, q], q0) for q0 in self.states)

            # do backward induction
            self.path[T-1] = np.argmax(self.max_proba[T-1, :])
            for t in range(T-2, -1, -1):
                self.path[t] = self.max_index[t, int(self.path[t+1])]
            
        elif self.hmm_type == 'rescaled':
            self.scale_factor_viterbi = np.zeros(T)
            
            # Initialize base cases (t == 0)
            self.max_proba[0, :] = self.pi * self.b[0, :]
            self.scale_factor_viterbi[0] = self.max_proba[0, :].sum()
            self.max_proba[0, :] = self.max_proba[0, :] / self.scale_factor_viterbi[0]
         
            # Run Viterbi for t > 0
            for t in range(1, T):
                for q in self.states:
                    (self.max_proba[t, q], self.max_index[t-1, q]) = max((self.max_proba[t-1, q0] * self.a[q0, q] * self.b[t, q], q0) for q0 in self.states)
                self.scale_factor_viterbi[t] = self.max_proba[t, :].sum()
                self.max_proba[t, :] = self.max_proba[t, :] / self.scale_factor_viterbi[t]

            # do backward induction
            self.path[T-1] = np.argmax(self.max_proba[T-1, :])
            for t in range(T-2, -1, -1):
                self.path[t] = self.max_index[t, int(self.path[t+1])]
        else:
            raise RuntimeError("unknow type allowed log-scale or rescaled")
        return
    
    def plot_cluster(self, data_name, data, classification_name, classification_set, prefix, suffix, show_cross=False):
        plt.figure()
        for i in range(self.K):
            cluster = data[classification_set==i, :]
            plt.scatter(cluster[:, 0], cluster[:, 1], color=COLORS[i])
            plot_ellipse(self.mu[i,:][:,None], np.linalg.inv(np.asmatrix(self.sigma[i,:,:])), \
                -2*np.log(ALPHA), color=COLORS[i])
            if show_cross:
                plt.scatter(self.mu[i,0], self.mu[i,1], color="k", marker="+", lw=20)
        plt.xlim(XLIM)
        plt.ylim(YLIM)
        plt.title('{} classification on {}'.format(classification_name, data_name))
        utils.save_figure(plt, prefix, suffix )
        return
            