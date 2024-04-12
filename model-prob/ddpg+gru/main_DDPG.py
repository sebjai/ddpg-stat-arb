# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
#plt.style.use('paper.mplstyle')

from MR_env import MR_env
from DDPG import DDPG
from gru_pred import gru_pred

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env = MR_env(S_0 = 1, kappa = 5, sigma = 0.1, theta = [0.9, 1, 1.1 ],
             dt = 0.1, T = 21.1, 
             I_max = 10, lambd = 0.05)

gru = gru_pred(T = env.T, dt = env.dt, learning_rate = 0.001,
                 seq_length = 10, n_ahead = 1, 
                 dropout_rate = 0, kappa = env.kappa, sigma = env.sigma, batch_size = 16)

ddpg = DDPG(env, gru = gru, I_max = 10,
            gamma = 0.999, 
            lr=1e-3,
            name="test" )
   
ddpg.train(n_iter=10_000, n_plot=100, n_iter_Q=5, n_iter_pi=5, mini_batch_size=1)
# %%
import torch

# Your original tensor
original_tensor = torch.randn(1000, 201)

# Define the sample size
sample_size = 50

# Sample a random index
num_samples = original_tensor.shape[0] - sample_size + 1
start_index = torch.randint(0, num_samples, (1,), dtype=torch.long).item()

# Extract the sample using the sampled index
sample = original_tensor[start_index:start_index+sample_size]

print("Shape of sample:", sample)  # Shape will be (50, 201)

# %%
