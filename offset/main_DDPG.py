# -*- coding: utf-8 -*-

import os, sys
sys.path.insert(0, os.path.abspath(".."))

import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

from offset import DDPG, offset_env

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#%%
env = offset_env.offset_env(T=1/12, sigma=0.5, kappa=0.03, eta = 0.05, xi=0.1,
                 c=0.25, S0=2.5, R=5, pen=2.5)

ddpg = DDPG.DDPG(env,
            gamma = 0.999, 
            lr=1e-3,
            name="test" )
 
#%%    
ddpg.train(n_plot=50, n_iter_Q=5, n_iter_pi=5)