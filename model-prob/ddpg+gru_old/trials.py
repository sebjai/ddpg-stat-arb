# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
#plt.style.use('paper.mplstyle')

from MR_env import MR_env
from DDPG import DDPG
from gru_pred import gru_pred

import os
import cProfile
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env = MR_env(S_0 = 1, kappa = 5, sigma = 0.1, theta = [0.9, 1, 1.1 ],
             dt = 0.1, T = 10.1, 
             I_max = 10, lambd = 0.05)

gru = gru_pred(T = env.T, dt = env.dt, learning_rate = 0.001,
                 seq_length = 10, n_ahead = 1, 
                 dropout_rate = 0, kappa = env.kappa, sigma = env.sigma, batch_size = 16)

ddpg = DDPG(env, gru = gru, I_max = 10,
            gamma = 0.999, 
            lr=1e-3,
            name="test" )
#%%
env.N 
#%%
batch = 64
theta = env.teta_per_batch(batch, env.N , 'MC')
s = torch.zeros(batch, env.N + 1)

theta[:,-1]
s, i = env.Simulate_know_theta(1, 0, theta, en=12, batch_size= batch)
s.shape


#%%
import torch
s = torch.zeros(batch, env.N + 1)
i = torch.zeros(batch, env.N + 1)

for j in range():
s[j, ...], i[j, ...] = env.Simulate_know_theta(1, 0, theta, en=12, batch_size= batch)



#%%
S = torch.zeros(batch, env.N)
for i in range(101 - 1):
    S[i, ...] = env.step(i, s, i, 0, theta[i])

S

#%%
theta.shape, s.shape
#%%
#s.shape, theta.shape
es, pr = ddpg.get_theta(s)
# %%
es.shape, pr.shape
# %%
pr.shape
# %%
s.shape