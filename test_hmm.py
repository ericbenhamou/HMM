# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:35:54 2017

@author: eric.benhamou
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
from hmm import Hmm

DPI = 300 #figure resolution

#make matplot interactive to avoid hanging figure
plt.ion()
utils.delete_all_png_file()


#Q1
def compare_recusions():
    # Compare the two version in terms of efficiency
    start_time = time.time()
    hmm = Hmm('rescaled')
    hmm.compute_proba()
    hmm.EM()
    print("rescaled version run in %.6s seconds " %(time.time() - start_time))
    
    start_time = time.time()
    hmm2 = Hmm('log-scale')
    hmm2.compute_proba()
    hmm2.EM()
    print("log-scale version run in %.6s seconds " %(time.time() - start_time))

    print('difference cond_proba ' %  np.min(hmm.cond_proba - hmm2.cond_proba), np.max(hmm.cond_proba - hmm2.cond_proba))
    print('difference joined_cond_proba' % np.min(hmm.joined_cond_proba - hmm2.joined_cond_proba), np.max(hmm.joined_cond_proba - hmm2.joined_cond_proba))
    return

#Q1
print( '\n****************** Q1 ******************')
compare_recusions()

#Q2
print( '\n****************** Q2 ******************')
hmm = Hmm('rescaled')
hmm2 = Hmm('log-scale') 
hmm.compute_proba(hmm.test_data)
hmm.plot_proba(100, 'conditional proba with initial parameters on test', 'q02', '1' )
    
#Q3 & Q4 
print( '\n****************** Q4 ******************')
hmm.EM(True)
hmm.print_parameters()

# Q5
print( '\n****************** Q5 ******************')
hmm.plot_likelihood('q05', '1')

# Q6
print( '\n****************** Q6 ******************')
hmm.print_loglikelihoods_table()

#Q7-Q8
def plot_Viterbi_path(hmm):
    hmm.compute_proba()
    hmm.EM()
    hmm.compute_viterbi_path(hmm.train_data)
    hmm.plot_most_likely_state( hmm.path, len(hmm.path), 'Most likely state (Viterbi) on train', 'q07', '1')
    hmm.plot_cluster('train', hmm.train_data, 'viterbi', hmm.path, 'q08', '1')
    return

def compare_models(hmm,hmm2):
    hmm.EM()
    hmm.compute_viterbi_path(hmm.train_data)
    hmm2.EM()
    hmm2.compute_viterbi_path(hmm2.train_data)
    if np.min(hmm.path - hmm2.path) == 0 and np.max(hmm.path - hmm2.path)==0:
        print('No difference between models: looking good!')
    else:
        print('Most likely there is a bug, please investigate!')
    return
    
print( '\n****************** Q7-Q8 ******************')
plot_Viterbi_path(hmm)
compare_models(hmm,hmm2)
#compare with log scale
#and check there is no difference

#Q9
print( '\n****************** Q9 ******************')
hmm.compute_proba(hmm.test_data)
hmm.plot_proba(100, 'Conditional proba with EM parameters on Test', 'q09', '1')

#Q10
print( '\n****************** Q10 ******************')
hmm.plot_most_likely_state(np.argmax(hmm.cond_proba[:100,:], 1), 100, 'Most likely state (cond Proba) on Test', 'q10', '1')
hmm.plot_cluster('Test', hmm.test_data, 'Cond proba', np.argmax(hmm.cond_proba,1), 'q10', '2')


#Q11
print( '\n****************** Q11 ******************')
hmm.compute_viterbi_path(hmm.test_data)
hmm.plot_most_likely_state( hmm.path, 100, 'Most likely state (Viterbi) on Test', 'q11', '1')
hmm.plot_cluster('Test', hmm.test_data, 'viterbi', hmm.path, 'q11', '2')

#put a pause to make sure we get the figure set up when running in command line
try:
    plt.pause(5)
except:
    print('failed in plt.plause')
#wait for the user 
    
os.system("pause")
