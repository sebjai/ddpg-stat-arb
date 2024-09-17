# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt

from MR_env_ddpg import MR_env
from DDPG_new import DDPG

import os
import cProfile
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env = MR_env(S_0 = 1 , kappa = 5, sigma = 0.3, theta =[0.9, 1, 1.1],
             dt = 0.2, T = 40, 
             I_max = 10, lambd = 0.05)

ddpg = DDPG(env, gru = None, I_max = 10,
            gamma = 0.999, 
            lr= 0.001,
            n_nodes=20, n_layers=5, 
            name="test" )

# %%        
ddpg.train(n_iter=10_000, n_iter_Q = 1, n_iter_pi = 5, n_plot=100, mini_batch_size=512)

#%%
import torch
ddpg.pi['net'].load_state_dict(torch.load('pi.pth'))
#ddpg.gru['net'].load_state_dict(torch.load('gru_layer.pth'))
r, S, I = ddpg.run_strategy(N=2000)
#ddpg.plot_policy()
import numpy as np
np.save('S.npy', S)
np.save('I.npy', I)
#%%