# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
#plt.style.use('paper.mplstyle')

from MR_env import MR_env
from DDPG_new import DDPG
from gru_pred import gru_pred

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env = MR_env(S_0 = 1, kappa = 5, sigma = 0.1, theta = [0.9, 1, 1.1 ],
             dt = 1, T = 11, # dt = 0.5, T = 10.1
             I_max = 10, lambd = 0.05)

gru = gru_pred(T = env.T, dt = env.dt, learning_rate = 0.001,
                 seq_length = 10, n_ahead = 1, 
                 dropout_rate = 0, kappa = env.kappa, sigma = env.sigma, batch_size = 16)

ddpg = DDPG(env, gru = gru, I_max = 10,
            gamma = 0.999, 
            lr=1e-3,
            n_nodes=20, n_layers=6, 
            name="test" )

# %%        
ddpg.train(n_iter=10_000, n_iter_Q = 10, n_iter_pi = 50, n_plot=10, mini_batch_size=256)

# %%
