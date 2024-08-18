# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt

from MR_env_ddpg import MR_env
from DDPG_new import DDPG

import os
import cProfile
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env = MR_env(S_0 = 1 , kappa = [3, 7], sigma = 0.3, theta =[0.9, 1, 1.1],
             dt = 0.2, T = 40, 
             I_max = 10, lambd = 0.05)

ddpg = DDPG(env, gru = None, I_max = 10,
            gamma = 0.999, 
            lr= 0.001,
            n_nodes=20, n_layers=6, 
            name="test" )

# %%      z  
ddpg.train(n_iter=10, n_iter_Q = 1, n_iter_pi = 5, n_plot=5, mini_batch_size=512)
# %%
#cProfile.run('ddpg.train(n_iter=1000, n_iter_Q = 1, n_iter_pi = 1, n_plot=500, mini_batch_size=12)')
# %%

#import torch
#torch.save(ddpg.pi['net'].state_dict(), 'pi.pth')
#torch.save(ddpg.Q_main['net'].state_dict(), 'Q.pth')

# %%
import torch
ddpg.pi['net'].load_state_dict(torch.load('pi.pth'))
#ddpg.gru['net'].load_state_dict(torch.load('gru_layer.pth'))
r, S, I = ddpg.run_strategy(N=2000)
#ddpg.plot_policy()
import numpy as np
np.save('S.npy', S)
np.save('I.npy', I)
#%%
I.shape



# %%
import numpy as np
num_it = 500
num_steps=  2000
#r = np.load('r.npy')

r = np.zeros((num_it, num_steps))
S = np.zeros((num_it, num_steps-2))
I = np.zeros((num_it, num_steps-2))
theta_post = np.zeros((num_it, num_steps, 3))
for i in range(num_it):
    r[i, ...], S[i, ...], I[i, ...] = ddpg.run_strategy(N=num_steps-2, no_plots=True)
np.save('r_1.npy', r)
np.save('S_1.npy', S)
np.save('I_1.npy', I)

# %%
import seaborn as sns
sns.histplot(r[:, :-12].sum(axis=1), bins = 51, kde=True)
plt.title('Histogram of rewards for 1000 episodes')
plt.axvline (r[:, :-12].sum(axis=1).mean(), color='r', linestyle='-', label = 'mean')
plt.axvline (r[:, :-12].sum(axis=1).mean() + r[:, :-12].sum(axis=1).std(), color='r', linestyle='--', label = 'std')
plt.axvline (r[:, :-12].sum(axis=1).mean() - r[:, :-12].sum(axis=1).std(), color='r', linestyle='--')
plt.axvline (r[:, :-12].sum(axis=1).mean() + 2*r[:,: -12].sum(axis=1).std(), color='r', linestyle='-.')
plt.axvline (r[:, :-12].sum(axis=1).mean() - 2*r[:,: -12].sum(axis=1).std(), color='r', linestyle='-.', label = r'2$\times$std')
plt.axvline(np.median(r[:, :-12].sum(axis=1)), color='b', linestyle='-.', label='median')
plt.axvline(0, color='g')
plt.legend()
plt.show()

t = np.arange(num_it - 12)
for i in range(1, num_it):
    plt.plot(np.cumsum(r[i, :- 12], axis = 0), linewidth=1, alpha=0.5)
plt.plot(np.cumsum(r[0, :- 12], axis = 0), color='k', linewidth=1)
plt.title('Cumulative rewards for 1000 episodes')
plt.xlabel('Time')

plt.show()
# %%
