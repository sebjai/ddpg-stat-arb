# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
#plt.style.use('paper.mplstyle')

from MR_env_ddpg import MR_env
from DDPG_new import DDPG
from gru_pred import gru_pred
#from main_gru_pred import model

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#kappa=5, sigma=0.22, dt=0.2


model = gru_pred(T=40, 
                 learning_rate = 0.001,
                 seq_length=10, n_ahead=1, 
                 dropout_rate=0, kappa=5, sigma=0.3, dt=0.2)

env = MR_env(S_0 = model.env.S_0 , kappa = model.env.kappa, sigma = model.env.sigma, theta = model.env.theta,
             dt = model.env.dt, T = model.env.T, 
             I_max = 10, lambd = 0.05)

#gru = gru_pred(T = env.T, dt = env.dt, learning_rate = 0.001,
#                 seq_length = 10, n_ahead = 1, 
#                 dropout_rate = 0, kappa = env.kappa, sigma = env.sigma, batch_size = 16)

ddpg = DDPG(env, gru = model, I_max = 10,
            gamma = 0.999, 
            lr= 0.001,
            n_nodes=16, n_layers=5, 
            name="test" )

# %%        
#ddpg.train(n_iter=10_000, n_iter_Q = 1, n_iter_pi = 5, n_plot=100, mini_batch_size=512)

#import torch
#torch.save(ddpg.pi['net'].state_dict(), 'pi.pth')
#torch.save(ddpg.Q_main['net'].state_dict(), 'Q.pth')
#%%
import torch
ddpg.pi['net'].load_state_dict(torch.load('pi.pth'))
r = ddpg.run_strategy(N=1000)
ddpg.plot_policy()
# %%
import numpy as np
num_it = 500
num_steps=  1002
r = np.load('r.npy')

#r = np.zeros((num_it, num_steps))
#for i in range(num_it):
#    r[i, ...] = ddpg.run_strategy(N=num_steps - 2, no_plots=  True)
# np.save('r.npy', r)
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
    self = ddpg
    NS = 101
    S = torch.linspace(self.env.S_0 - 3*self.env.inv_vol,
                       self.env.S_0 + 3*self.env.inv_vol,
                       NS)
    NI = 51
    I = torch.linspace(-self.I_max/2, self.I_max/2, NI)
    Sm, Im = torch.meshgrid(S, I, indexing='ij')
    
    def plot(a, ax):
        cs = ax.contourf(Sm.squeeze().numpy(), Im.squeeze().numpy(), a.numpy(),
                         levels=np.linspace(-self.I_max, self.I_max, 21),
                         cmap='RdBu')
        ax.axvline(self.env.S_0, linestyle='--', color='g')
        ax.axvline(self.env.S_0-2*self.env.inv_vol, linestyle='--', color='k')
        ax.axvline(self.env.S_0+2*self.env.inv_vol, linestyle='--', color='k')
        ax.axhline(0, linestyle='--', color='k')
        ax.axhline(self.I_max/2, linestyle='--', color='k')
        ax.axhline(-self.I_max/2, linestyle='--', color='k')
        return cs

    Sm = Sm.unsqueeze(-1)
    Im = Im.unsqueeze(-1)
    ones = torch.ones(Sm.shape)
    pi_all = [0.8, 0.2, 0.2]

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)
    for i in range(len(pi_all)):
        for j in range(len(pi_all)):
            pi1 = pi_all[i]
            pi2 = pi_all[j]
            pi3 = 1 - pi1 - pi2
            theta_estim = torch.cat((pi1 * ones, pi2 * ones, pi3 * ones), axis=-1)
            X = torch.cat(((Sm / self.env.S_0 - 1.0),
                           (Im / self.I_max),
                           theta_estim), axis=-1)
            a = ddpg.pi['net'](X).detach().squeeze()
            cs = plot(a, ax[i, j])
            ax[j, i].set_ylabel(f"$\pi${j + 1} " if i == 0 else None)
    
    # Create an axis for the colorbar on the right side of the figure
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(cs, cax=cbar_ax)
    cbar.set_ticks(np.linspace(-self.I_max, self.I_max, 11))
    cbar.ax.set_ylabel('Action')

    #plt.tight_layout(rect=[0, 0, 0.85, 1])
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Price")
    plt.ylabel("Inventory")
    plt.show()

#%%

ddpg.plot_policy()
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
