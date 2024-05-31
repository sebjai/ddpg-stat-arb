# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
#plt.style.use('paper.mplstyle')

from MR_env import MR_env
from DDPG_new import DDPG
from gru_pred import gru_pred

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#kappa=5, sigma=0.22, dt=0.2

env = MR_env(S_0 = 1, kappa = 5, sigma = 0.22, theta = [0.9, 1, 1.1 ],
             dt = 0.2, T = 11, 
             I_max = 10, lambd = 0.05)

gru = gru_pred(T = env.T, dt = env.dt, learning_rate = 0.001,
                 seq_length = 10, n_ahead = 1, 
                 dropout_rate = 0, kappa = env.kappa, sigma = env.sigma, batch_size = 16)

ddpg = DDPG(env, gru = gru, I_max = 10,
            gamma = 0.999, 
            lr= 0.002,
            n_nodes=16, n_layers=5, 
            name="test" )

# %%        
ddpg.train(n_iter=10_000, n_iter_Q = 5, n_iter_pi = 5, n_plot=100, mini_batch_size=512)

# %%
