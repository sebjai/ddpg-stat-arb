# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
#plt.style.use('paper.mplstyle')

from MR_env_ddpg import MR_env
from DDPG_new import DDPG
#from gru_pred import gru_pred
#from main_gru_pred import model

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#kappa=5, sigma=0.22, dt=0.2


#model = gru_pred(T=40, 
#                 learning_rate = 0.001,
#                 seq_length=10, n_ahead=1, 
#                 dropout_rate=0, kappa=5, sigma=0.3, dt=0.2)

env = MR_env(S_0 = 1 , kappa = 5, sigma = 0.3, theta =[0.9, 1, 1.1],
             dt = 0.2, T = 40, 
             I_max = 10, lambd = 0.05)

#gru = gru_pred(T = env.T, dt = env.dt, learning_rate = 0.001,
#                 seq_length = 10, n_ahead = 1, 
#                 dropout_rate = 0, kappa = env.kappa, sigma = env.sigma, batch_size = 16)

ddpg = DDPG(env, gru = None, I_max = 10,
            gamma = 0.999, 
            lr= 0.001,
            n_nodes=20, n_layers=5, 
            name="test" )

# %%        
ddpg.train(n_iter=10_000, n_iter_Q = 1, n_iter_pi = 5, n_plot=100, mini_batch_size=512)

#import torch
#torch.save(ddpg.pi['net'].state_dict(), 'pi.pth')
#torch.save(ddpg.Q_main['net'].state_dict(), 'Q.pth')
#%%
import torch
ddpg.pi['net'].load_state_dict(torch.load('pi.pth'))
r, S, I = ddpg.run_strategy(N=1000)
ddpg.plot_policy()
# %%
r, S, I = ddpg.run_strategy_rolling(N=1000)

# %%
# t = N#
a = env.dt*np.arange(0, env.N)/env.T
t = a[:-(ddpg.seq_length + 2)]
plt.figure(figsize=(5,5))
n_paths = 3
fig, axs = plt.subplots(n_paths, 1, figsize=(10, 15))
for i in range(0,len(t),10):
    plt.plot((S[:,i:i+50]).squeeze(0).numpy(), label='Stock price')
    #plt.set_ylabel('Stock price')
    #plt.set_xlabel('Time')
    plt.legend(loc='upper left')
    #ax2 = axs[i].twinx()
    plt.plot((I[:,i:i+50]).squeeze(0), color='red', label='Inventory')
    #plt.set_ylabel('Inventory')
    plt.legend(loc='upper right')
plt.title("Stock price and Inventory")
plt.show()
# %%
S[:,i:i+100].squeeze(0).numpy()


#%%
import numpy as np
num_it = 500
num_steps=  1002
#r = np.load('r.npy')

r = np.zeros((num_it, num_steps))
for i in range(num_it):
    r[i, ...] = ddpg.run_strategy(N=num_steps - 2, no_plots=  True)
np.save('r.npy', r)
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
#%%

#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
from MR_env_ddpg import MR_env
from DDPG_new import DDPG
from gru_pred import gru_pred
import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_policy(ddpg, name=""):
        NS = 101
        I_max = 10
        S = torch.linspace( env.S_0 - 3* env.inv_vol,
                            env.S_0 + 3* env.inv_vol,
                           NS)
        NI = 51
        I = torch.linspace(- I_max,  I_max, NI)
        Sm, Im = torch.meshgrid(S, I,indexing='ij')
        def plot(a, ax):
            cs = ax.contourf(Sm.squeeze().numpy(), Im.squeeze().numpy(), a.numpy(),
                              levels=np.linspace(- I_max,  I_max, 21),
                              cmap='RdBu')
            ax.axvline( env.S_0, linestyle='--', color='g')
            ax.axvline( env.S_0-2* env.inv_vol, linestyle='--', color='k')
            ax.axvline( env.S_0+2* env.inv_vol, linestyle='--', color='k')
            ax.axhline(0, linestyle='--', color='k')
            ax.axhline( I_max/2, linestyle='--', color='k')
            ax.axhline(- I_max/2, linestyle='--', color='k')
            # cbar = fig.colorbar(cs, ax=ax, shrink=0.9)
            # cbar.set_ticks(np.linspace(- I_max,  I_max, 11))
            # cbar.ax.set_ylabel('Action')
        Sm = Sm.unsqueeze(-1)
        Im = Im.unsqueeze(-1)
        ones = torch.ones(Sm.shape)
        pi_all = [0.25, 0.5, 0.75]
        fig, ax = plt.subplots(3,3, figsize=(8,8), sharex=True,sharey=True)
        for i in range(len(pi_all)):
            for j in range(len(pi_all)):
                pi1 = pi_all[i]
                pi2 = pi_all[j]
                pi3 = 1-pi1-pi2
                theta_estim = torch.cat((pi1*ones, pi2*ones, pi3*ones),axis=-1)
                X = torch.cat(((Sm/ env.S_0-1.0),
                               (Im/ I_max  ),
                               theta_estim), axis=-1)
                a =  ddpg.pi['net'](X).detach().squeeze()
                plot(a, ax[i,j])
        plt.tight_layout()
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Price")
        plt.ylabel("Inventory")
        plt.show()

#%%

plot_policy(ddpg)
# %%

S = ddpg.obtain_data(mini_batch_size   =  1, N = 1000, train = False)
# %%
import numpy as np
np.save('S.npy', S)
# %%

S, _, theta_true = env.Simulate(s0=1, 
                                             i0=0, 
                                             model = 'MC', batch_size=1, ret_reward = False, 
                                             I_p = 0, N = 200)
S
# %%
plt.plot(S.squeeze())
plt.plot(theta_true.squeeze())
# %%
r = np.load('r.npy')
x = np.zeros((r.shape[0], r.shape[1]-12))
num_it = r.shape[0]
t = np.arange(num_it - 12)
for i in range(1, num_it):
    x[i, ... ] = np.cumsum(r[i, :- 12], axis = 0)
    plt.plot(np.cumsum(r[i, :- 12], axis = 0), linewidth=1, alpha=0.5)
plt.plot(np.cumsum(r[0, :- 12], axis = 0), color='k', linewidth=1)
plt.title('Cumulative rewards for 1000 episodes')
plt.xlabel('Time')

plt.show()
qtl = np.quantile(x, [0.01, 0.5, 0.99], axis = 0)
t = np.arange(x.shape[1])*0.038
for i in range(1, 100):
    plt.plot(t,np.cumsum(r[i, :- 12], axis = 0), linewidth=1, alpha=0.2)
plt.fill_between(t, qtl[0], qtl[2], color='b', alpha=0.3)
plt.plot(t,qtl[1], color='b', linewidth=1)
plt.title('Cumulative rewards for 1.000 trading episodes using DDPG')
plt.xlabel('Time')
plt.ylabel('Cumulative reward')
plt.savefig('cumulative_reward.png')

plt.show()
# %%
