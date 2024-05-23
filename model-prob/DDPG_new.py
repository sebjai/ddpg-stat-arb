# -*- coding: utf-8 -*-
"""
Created on Mon May  13 13:39:56 2024

@author: macrandrea
"""
from MR_env import MR_env as Environment
from gru_pred import gru_pred as RNN
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from datetime import datetime
import scipy.linalg as linalg 
import random
import pdb
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ANN(nn.Module):

    def __init__(self, n_in, n_out, nNodes, nLayers, 
                 activation='silu', out_activation=None,
                 scale = 1):
        super(ANN, self).__init__()
        
        self.prop_in_to_h = nn.Linear(n_in, nNodes)

        # modulelist informs pytorch that these have parameters 
        # that you need to compute gradients of
        self.prop_h_to_h = nn.ModuleList(
            [nn.Linear(nNodes, nNodes) for i in range(nLayers-1)])

        self.prop_h_to_out = nn.Linear(nNodes, n_out)
        
        if activation == 'silu':
            self.g = nn.SiLU()
        elif activation == 'relu':
            self.g = nn.ReLU()
        elif activation == 'sigmoid':
            self.g= torch.sigmoid()
        
        self.out_activation = out_activation
        self.scale = scale

    def forward(self, x):

        # input into  hidden layer
        h = self.g(self.prop_in_to_h(x))

        for prop in self.prop_h_to_h:
            h = self.g(prop(h))

        # hidden layer to output layer
        y = self.prop_h_to_out(h)

        if self.out_activation == 'tanh':
            y = torch.tanh(y)
            
        y = self.scale * y 

        return y

class DDPG():

    def __init__(self, env: Environment, gru: RNN , I_max=10, 
                 gamma=0.99,  
                 n_nodes=36, n_layers=6, 
                 lr=1e-3, sched_step_size = 50, tau=0.001,
                 name=""):

        self.gamma = gamma
        self.I_max = I_max
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.name =  name
        self.sched_step_size = sched_step_size
        self.lr = lr
        self.seq_length = gru.seq_length
        self.n_ahead = gru.n_ahead
        self.lg = nn.Softmax(dim=1)
        self.tau = tau
        
        self.__initialize_NNs__()
        
        self.S = []
        self.I = []
        self.q = []
        self.r = []
        self.epsilon = []
        
        self.Q_loss = []
        self.pi_loss = []
        self.gru = gru
        self.gru.model.load_state_dict(torch.load('model.pth')) # load the model
        
        self.env = env

    def __initialize_NNs__(self):
        
        # policy approximation
        #
        # features = S, I, theta_estim_1, theta_estim_2, theta_estim_3
        #
        self.pi = {'net': ANN(n_in=5, 
                              n_out=1, 
                              nNodes=self.n_nodes, 
                              nLayers=self.n_layers,
                              out_activation='tanh',
                              scale=self.I_max)}
        
        self.pi['optimizer'], self.pi['scheduler'] = self.__optim_and_scheduler__(self.pi)        
        
        # Q - function approximation
        #
        # features = S, I, a=Ip, theta_estim_1, theta_estim_2, theta_estim_3
        #
        self.Q_main = {'net' : ANN(n_in=6, 
                                  n_out=1,
                                  nNodes=self.n_nodes, 
                                  nLayers=self.n_layers) }

        self.Q_main['optimizer'], self.Q_main['scheduler'] = self.__optim_and_scheduler__(self.Q_main)
        self.Q_target = copy.deepcopy(self.Q_main)
        
    def __optim_and_scheduler__(self, net):
        
        optimizer = optim.AdamW(net['net'].parameters(),
                                lr=self.lr)
                    
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.sched_step_size,  gamma=0.99)
    
        return optimizer, scheduler
    
    def __stack_state__(self, S, I, theta_estim):

        return torch.cat((
            (S/self.env.S_0-1.0).unsqueeze(-1), 
            (I/self.I_max  ).unsqueeze(-1), 
            theta_estim), axis=1)

    
    def __grab_mini_batch__(self, mod_type = 'MC', mini_batch_size = 256):
        
        t = torch.rand((mini_batch_size))*self.env.N
        S, I, theta_true = self.env.Randomize_Start(type_mod = mod_type, batch_size = mini_batch_size)
        # da qui escono già come snippets

        return t, S, I, theta_true
    
    def create_snippets(self, x, theta=None):
        
        x_cp = x.T
        if theta is not None:
            theta_cp = theta.T
        x_cp = x
        Z = torch.zeros((self.seq_length, 
                         x_cp.shape[0]-self.seq_length, #-self.n_ahead
                         x_cp.shape[1]))
        
        Y = torch.zeros((x_cp.shape[0]-self.seq_length, #-self.n_ahead
                         x_cp.shape[1]))        
        
        for i in range(x_cp.shape[0]-self.seq_length):#-self.n_ahead
            Z[:,i,:] = x_cp[i:i+self.seq_length,:]

            if theta is None:
                Y[i,:] = x_cp[i+self.seq_length+(self.n_ahead-1),:]
            else:
                Y[i,:] = theta_cp[i+self.seq_length,:]
        
        return Z.transpose(0,1), Y

    def obtain_data(self, mini_batch_size = 256, N =  12, train = True):
            
        S, _, theta_true = self.env.Simulate(s0=self.env.S_0 + 3*self.env.inv_vol*torch.randn(mini_batch_size, ), 
                                             i0=self.I_max * (2*torch.rand(mini_batch_size)-1), 
                                             model = 'MC', batch_size=mini_batch_size, ret_reward = False, 
                                             I_p = 0, N = N)

        I = self.I_max * (2*torch.rand(mini_batch_size, S.shape[1])-1)
        
        if train == True:   

            snip, _ = self.create_snippets(S[:,:self.seq_length+self.n_ahead].T)

            theta_pred = self.lg(self.gru.model(snip.transpose(0,2)).detach().squeeze())

            snip_p, _ = self.create_snippets(S[:,1:self.seq_length+self.n_ahead +1].T)
            
            theta_pred_p = self.lg(self.gru.model(snip_p.transpose(0,2)).detach().squeeze())

            return S, I, theta_pred, theta_pred_p
        
        else:

            return S
        

    def make_reward(self, batch_S, batch_S_p, batch_I, I_p):

        q = I_p-batch_I

        r = I_p*(batch_S_p-batch_S) - self.env.lambd*torch.abs(q)

        return r
    
    def get_theta(self, S):

        snip, _ = self.create_snippets(S[:,:self.seq_length+self.n_ahead].T)

        theta_pred = self.lg(self.gru.model(snip.transpose(0,2)).detach())

        return theta_pred

    def Update_Q(self, mini_batch_size=256, epsilon=0.02, n_iter_Q=1):

        for i in range(n_iter_Q):
            
            batch_S, batch_I, theta_estim, theta_estim_p = self.obtain_data(mini_batch_size, N = self.env.N+1)

            self.Q_main['optimizer'].zero_grad()
    
            # concatenate states
            X = self.__stack_state__(batch_S[:,self.seq_length-1], 
                                     batch_I[:,self.seq_length-1],
                                     theta_estim) 
            
            I_p = self.pi['net'](X).detach() * torch.exp(-0.5*epsilon**2+epsilon*torch.randn(batch_S.shape[0],1))
    
            Q = self.Q_main['net']( torch.cat((X, I_p/self.I_max),axis=-1) )
            
            # step in the environment
            r = self.make_reward(batch_S[:, self.seq_length-1], 
                                 batch_S[:, self.seq_length], 
                                 batch_I[:, self.seq_length-1], I_p)

            # compute the Q(S', a*)
            X_p = self.__stack_state__(batch_S[:, self.seq_length], I_p.squeeze(-1), theta_estim_p)   

            # optimal policy at t+1
            I_pp = self.pi['net'](X_p).detach()
        
            # compute the target for Q
            target = r + self.gamma * self.Q_target['net'](torch.cat((X_p, I_pp/self.I_max), axis=1))
        
            loss = torch.mean((target.detach() - Q)**2)
        
            loss.backward()
    
            # perform step using those gradients
            self.Q_main['optimizer'].step()                
            self.Q_main['scheduler'].step() 
        
            self.Q_loss.append(loss.item())

            self.soft_update( self.Q_main['net'], self.Q_target['net'])
        
    def soft_update(self, main, target):
    
        for param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
    def Update_pi(self, mini_batch_size=256, epsilon=0.02, n_iter_pi=10):

        for i in range(n_iter_pi):
            
            batch_S, batch_I, theta_estim, _ = self.obtain_data(mini_batch_size, N = self.env.N+1)

            self.pi['optimizer'].zero_grad()

            X = self.__stack_state__(batch_S[:,self.seq_length-1], 
                                     batch_I[:,self.seq_length-1],  
                                     theta_estim) 
        
            I_p = self.pi['net'](X)
            
            Q = self.Q_main['net']( torch.cat((X, I_p/self.I_max),axis=-1) )

            loss = -torch.mean(Q)# -
                
            loss.backward()
            
            self.pi['optimizer'].step()
            self.pi['scheduler'].step()
           
            self.pi_loss.append(loss.item())

    def train(self, n_iter=1_000, n_iter_Q= 1, n_iter_pi = 10, mini_batch_size=256, n_plot=100):
        C = 100
        D = 100
        
        if len(self.epsilon)==0:
            self.count=0

        for i in tqdm(range(n_iter)):

            epsilon = np.maximum(C/(D+self.count), 0.02)
            self.epsilon.append(epsilon)
            self.count += 1

            self.Update_Q(mini_batch_size=mini_batch_size, epsilon = epsilon, n_iter_Q= n_iter_Q )

            self.Update_pi(mini_batch_size=mini_batch_size, epsilon = epsilon, n_iter_pi = n_iter_pi)

            if np.mod(i+1,n_plot) == 0:
                
                self.loss_plots()
                self.run_strategy(name= datetime.now().strftime("%H_%M_%S"), N = 500)

    def moving_average(self, x, n):
        
        y = np.zeros(len(x))
        y_err = np.zeros(len(x))
        y[0] = np.nan
        y_err[0] = np.nan
        
        for i in range(1,len(x)):
            
            if i < n:
                y[i] = np.mean(x[:i])
                y_err[i] = np.std(x[:i])
            else:
                y[i] = np.mean(x[i-n:i])
                y_err[i] = np.std(x[i-n:i])
                
        return y, y_err                
            
    def loss_plots(self):
        
        def plot(x, label, show_band=True):

            mv, mv_err = self.moving_average(x, 100)
        
            if show_band:
                plt.fill_between(np.arange(len(mv)), mv-mv_err, mv+mv_err, alpha=0.2)
            plt.plot(mv, label=label, linewidth=1) 
            plt.legend()
            plt.ylabel('loss')
            plt.yscale('symlog')
        
        fig = plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plot(self.Q_loss, r'$Q$', show_band=False)
        
        plt.subplot(1,2,2)
        plot(self.pi_loss, r'$\pi$')
        
        plt.tight_layout()
        plt.show()

    def run_strategy(self, name= datetime.now().strftime("%H_%M_%S"), N = 12, no_plots = False):

        S = torch.zeros((1, N+2)).float()         
        I = torch.zeros((1, N+2)).float()
        r = torch.zeros((1, N+2)).float()
        theta_post = torch.zeros((1, N+2)).float()

        S = self.obtain_data(mini_batch_size   =  1, N = N, train = False)

        for t in range(N-self.seq_length):

            theta_post = self.get_theta(S[:, t:t+self.seq_length+1])

            X = self.__stack_state__(S[:, t+self.seq_length-1].T, I[:, t+self.seq_length-1].T, theta_post)

            I[:, t+self.seq_length] = self.pi['net'](X).reshape(-1).detach().unsqueeze(-1)

            r[:, t+1] = self.make_reward(S[:,t+self.seq_length-1], 
                                         S[:,t+self.seq_length],
                                         I[:,t+self.seq_length-1], 
                                         I[:,t+self.seq_length])

        I  =    I.detach().numpy()
        r =     r.detach().numpy()


        if no_plots == False:
            # t = N#
            a = self.env.dt*np.arange(0, N)/self.env.T
            t = a[:-(self.seq_length + 2)]
            plt.figure(figsize=(5,5))
            n_paths = 3

            #plt.figure(figsize=(10, 15))

            #plt.subplot(2,1, 1)
            fig, ax1 = plt.subplots()

            ax1.plot((S[:,self.seq_length:-2]).squeeze(0).numpy(), label = 'Stock price')
            ax1.set_ylabel('Stock price')
            ax1.set_xlabel('Time')
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            ax2.plot((I[:,self.seq_length:-2]).squeeze(0), color='red', label = 'Inventory')
            ax2.set_ylabel('Inventory')
            ax2.legend(loc='upper right')

            plt.title("Stock price and Inventory")
            plt.show()

            #plt.subplot(3,1, 2)
             #(t, I.squeeze(0).numpy(), 2, r"$I_t$")
            #plt.title("Inventory")

            #plt.subplot(2,1, 2)
            plt.plot(np.cumsum(r.squeeze(0)))
            plt.title("cumulative return")

            #plt.subplot(4,1, 4)
            #plt.hist(r.squeeze(0), bins=51)
            #plt.title("histogram of returns")
            plt.tight_layout()
        
            plt.savefig("path_"  +self.name + "_" + name + ".pdf", format='pdf', bbox_inches='tight')
            plt.show()

            return r
        if no_plots == True:
            return r    
    
    
    def plot_policy(self, name=""):
        NS = 101
        S = torch.linspace(self.env.S_0 - 5*self.env.inv_vol,
                           self.env.S_0 + 5*self.env.inv_vol,
                           NS)
        NI = 51
        I = torch.linspace(-self.I_max, self.I_max, NI)
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
        pi_all = [(0.9, 0.05, 0.05),  # pi1 high
                  (0.05, 0.9, 0.05),  # pi2 high
                  (0.05, 0.05, 0.9)]  # pi3 high

        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        for i in range(len(pi_all)):
            pi1, pi2, pi3 = pi_all[i]
            theta_estim = torch.cat((pi1 * ones, pi2 * ones, pi3 * ones), axis=-1)
            X = torch.cat(((Sm / self.env.S_0 - 1.0),
                           (Im / self.I_max),
                           theta_estim), axis=-1)
            a = self.pi['net'](X).detach().squeeze()
            cs = plot(a, ax[i])
            #ax[i].set_title(f"$\pi_{{{i+1}}}$ high")
            ax[0].set_title(f"$\pi_1$ high \n $\\theta = 0.9$")
            ax[1].set_title(f"$\pi_2$ high \n $\\theta = 1$")
            ax[2].set_title(f"$\pi_3$ high \n $\\theta = 1.1$")

        # Create an axis for the colorbar on the right side of the figure
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(cs, cax=cbar_ax)
        cbar.set_ticks(np.linspace(-self.I_max, self.I_max, 11))
        cbar.ax.set_ylabel('Action')

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Price")
        plt.ylabel("Inventory")
        plt.show()







