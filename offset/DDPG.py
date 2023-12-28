# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:39:56 2022

@author: sebja
"""

from offset_env import offset_env as Environment

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)

from tqdm import tqdm

import copy

import pdb

from datetime import datetime

class ANN(nn.Module):

    def __init__(self, n_in, n_out, nNodes, nLayers, 
                 activation='silu', out_activation=None):
        super(ANN, self).__init__()
        
        self.prop_in_to_h = nn.Linear(n_in, nNodes)

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

    def forward(self, x):

        # input into  hidden layer
        h = self.g(self.prop_in_to_h(x))

        for prop in self.prop_h_to_h:
            h = self.g(prop(h))

        # hidden layer to output layer
        y = self.prop_h_to_out(h)

        if self.out_activation is not None:
            for i in range(y.shape[-1]):
                y[...,i] = self.out_activation[i](y[...,i])
            
        return y


class DDPG():

    def __init__(self, env: Environment,  
                 gamma=0.9999,  
                 n_nodes=36, n_layers=3, 
                 lr=1e-3, sched_step_size = 100,
                 name=""):

        self.env = env
        self.gamma = gamma
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.name =  name
        self.sched_step_size = sched_step_size
        self.lr = lr
        
        self.__initialize_NNs__()
        
        self.t = []
        self.S = []
        self.X = []
        self.nu = []
        self.p = []
        self.r = []
        self.epsilon = []
        
        self.Q_loss = []
        self.pi_loss = []
        
        
    def __initialize_NNs__(self):
        
        # policy approximation
        #
        # features = t, S, X
        # out = nu, p
        #
        self.pi = {'net': ANN(n_in=3, 
                              n_out=2, 
                              nNodes=self.n_nodes, 
                              nLayers=self.n_layers,
                              out_activation=[lambda y : y, 
                                              torch.sigmoid])}
        
        self.pi['optimizer'], self.pi['scheduler'] = self.__get_optim_sched__(self.pi)        
        
        # Q - function approximation
        #
        # features = t, S, X, nu, p
        # out = Q
        self.Q_main = {'net' : ANN(n_in=5, 
                                  n_out=1,
                                  nNodes=self.n_nodes, 
                                  nLayers=self.n_layers) }

        self.Q_main['optimizer'], self.Q_main['scheduler'] = self.__get_optim_sched__(self.Q_main)
        
        self.Q_target = copy.deepcopy(self.Q_main)
        
        
    def __get_optim_sched__(self, net):
        
        optimizer = optim.AdamW(net['net'].parameters(),
                                lr=self.lr)
                    
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.sched_step_size,
                                              gamma=0.99)
    
        return optimizer, scheduler
        
    def __stack_state__(self, t, S, X):
        # normalization happens outside of stack state
        tS = torch.cat((t.unsqueeze(-1), 
                        S.unsqueeze(-1)), axis=-1)
        tSX = torch.cat((tS,
                         X.unsqueeze(-1)), axis=-1)
        return tSX
    
    
    def __grab_mini_batch__(self, mini_batch_size):
        t, S, X = self.env.randomize(mini_batch_size)
        return t, S, X
   
    def range_test(self, x, test='prob'):
        if test=='prob':
            if torch.amin(x) < 0 or torch.amax(x) > 1:
                print(torch.amin(x), torch.amax(x))

    def norm(self, k: torch.tensor, typ :str):
        
        norm = torch.zeros(k.shape)
        
        if typ == 'state':
            norm[...,0] = self.env.T
            norm[...,1] = self.env.S0
            norm[...,2] = self.env.X_max
            
        if typ == 'policy':
            norm[...,0] = self.env.nu_max
            norm[...,1] = 1.0
            
        return norm

    def normalize(self, k: torch.tensor, typ: str):
        '''
        possible types: "state" and "policy"
        '''
        norm = self.norm(k, typ)
            
        return k / norm

    def de_normalize(self, k: torch.tensor, typ: str):
        
        norm = self.norm(k, typ)
            
        return k * norm

    def Update_Q(self, n_iter = 10, mini_batch_size=256, epsilon=0.02):
        
        for i in range(n_iter): 
            
            t, S, X = self.__grab_mini_batch__(mini_batch_size)
            
            self.Q_main['optimizer'].zero_grad()
            
            # used for randomization
            nu_rand = torch.exp (epsilon*torch.randn((mini_batch_size,)))
            p_rand = 1.0/(1.0 + torch.exp(-epsilon*torch.randn((mini_batch_size,))))
            # H = torch.bernoulli(
            #     epsilon * torch.ones(mini_batch_size).to(torch.float32))
            H = 1.0* ( torch.rand(mini_batch_size) < epsilon)
            
            # concatenate states
            Y = self.__stack_state__(t, S, X)
            
            # normalize : Y (tSX)
            # get pi (policy)
            a = self.pi['net'](self.normalize(Y, 'state')).detach()

            # randomize policy and prob for exploration
            a[:,0] = a[:,0] * nu_rand
            a[:,1] = a[:,1] * p_rand * H + \
                (1 - (1 - a[:,1]) * p_rand) * (1-H)

            # get Q
            Q = self.Q_main['net']( torch.cat(( \
                                               self.normalize(Y, 'state'), \
                                               self.normalize(a, 'policy')), \
                                              axis=1) )

            # step in the environment
            Y_p, r = self.env.step(Y, a)
            
            ind_T = 1.0 * (torch.abs(Y_p[:,0] - self.env.T) <= 1e-6)

            # compute the Q(S', a*)
            # optimal policy at t+1
            a_p = self.pi['net'](self.normalize(Y_p, 'state')).detach()
            
            # compute the target for Q
            target = r.reshape(-1,1) + (1.0 - ind_T) * self.gamma * \
                self.Q_target['net'](torch.cat(( \
                                                self.normalize(Y_p, 'state'), \
                                                self.normalize(a_p, 'policy')), \
                                               axis=1))
                  
            loss = torch.mean((target.detach() - Q)**2)
            
            # compute the gradients
            loss.backward()

            # perform step using those gradients
            self.Q_main['optimizer'].step()                
            self.Q_main['scheduler'].step() 
            
            self.Q_loss.append(loss.item())
            
        self.Q_target = copy.deepcopy(self.Q_main)
        
    def Update_pi(self, n_iter = 10, mini_batch_size=256, epsilon=0.02):

        for i in range(n_iter):
            t, S, X = self.__grab_mini_batch__(mini_batch_size)
            
            self.pi['optimizer'].zero_grad()
            
            # concatenate states 
            Y = self.__stack_state__(t, S, X)

            a = self.pi['net'](self.normalize(Y, 'state'))
            
            Q = self.Q_main['net']( torch.cat(( \
                                               self.normalize(Y, 'state'),
                                               self.normalize(a, 'policy')),
                                              axis=1) )
            
            loss = -torch.mean(Q)
                
            loss.backward()
            
            self.pi['optimizer'].step()
            self.pi['scheduler'].step()
            
            self.pi_loss.append(loss.item())
            
    def train(self, n_iter=1_000, 
              n_iter_Q=10, 
              n_iter_pi=5, 
              mini_batch_size=256, 
              n_plot=100):
        
        self.run_strategy(1_000, name= datetime.now().strftime("%H_%M_%S"))
        self.plot_policy(name=datetime.now().strftime("%H_%M_%S"))

        C = 100
        D = 200
        
        if len(self.epsilon)==0:
            self.count=0
            
        for i in tqdm(range(n_iter)):
            
            epsilon = np.maximum(C/(D+self.count), 0.02)
            self.epsilon.append(epsilon)
            self.count += 1

             
            self.Update_Q(n_iter=n_iter_Q, 
                          mini_batch_size=mini_batch_size, 
                          epsilon=epsilon)
            
            self.Update_pi(n_iter=n_iter_pi, 
                           mini_batch_size=mini_batch_size, 
                           epsilon=epsilon)

            if np.mod(i+1,n_plot) == 0:
                
                self.loss_plots()
                self.run_strategy(1_000, name= datetime.now().strftime("%H_%M_%S"))
                # self.plot_policy()
                self.plot_policy(name=datetime.now().strftime("%H_%M_%S"))
                
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
        
    def run_strategy(self, nsims=10_000, name="", N = None):
        
        if N is None:
            N = self.env.N
        
        S = torch.zeros((nsims, N)).float()
        X = torch.zeros((nsims, N)).float()
        a = torch.zeros((nsims, 2, N-1)).float()
        r = torch.zeros((nsims, N-1)).float()


        S[:,0] = self.env.S0
        X[:,0] = 0
        
        ones = torch.ones(nsims)

        for k in range(N-1):
            # Y = self.__stack_state__(self.env.dt*t*ones ,S[:,t], X[:,t])
            Y = self.__stack_state__(self.env.t[k]* ones ,S[:,k], X[:,k])

            # normalize : Y (tSX)
            # get policy
            a[:,:,k] = self.pi['net'](self.normalize(Y, 'state'))

            # step in environment
            Y_p, r[:,k] = self.env.step(Y, a[:,:,k])
        
            # update subsequent state and inventory
            S[:, k+1] = Y_p[:,1]
            X[:, k+1] = Y_p[:,2]
            
        # print(torch.cat( (torch.amin(a[:,0,:], dim = 0).unsqueeze(-1), torch.amax(a[:,0,:].unsqueeze(-1), dim = 0)) , dim=-1))
        # print(a[:,0,:])

        S = S.detach().numpy()
        X  = X.detach().numpy()
        a = a.detach().numpy()
        r = r.detach().numpy()

        plt.figure(figsize=(8,5))
        n_paths = 3
        
        def plot(t, x, plt_i, title ):
            
            # print(x.shape)
            # pdb.set_trace()
            qtl= np.quantile(x, [0.05, 0.5, 0.95], axis=0)

            plt.subplot(2, 3, plt_i)
            plt.fill_between(t, qtl[0,:], qtl[2,:], alpha=0.5)
            plt.plot(t, qtl[1,:], color='k', linewidth=1)
            plt.plot(t, x[:n_paths, :].T, linewidth=1)
            
            # plt.xticks([0,0.5,1])
            plt.title(title)
            plt.xlabel(r"$t$")


        plot(self.env.t, (S), 1, r"$S_t$" )
        plot(self.env.t, X, 2, r"$X_t$")
        plot(self.env.t[:-1], np.cumsum(r, axis=1), 3, r"$r_t$")
        plot(self.env.t[:-1], a[:,0,:], 4, r"$\nu_t$")
        plot(self.env.t[:-1], a[:,1,:], 5, r"$p_t$")
        # plot(t, np.cumsum(r, axis=1), 3, r"$r_t$")

        plt.subplot(2, 3, 6)
        plt.hist(np.sum(r,axis=1), bins=51)
        #plt.set_title('PnL')


        plt.tight_layout()
        
       # plt.savefig("path_"  +self.name + "_" + name + ".pdf", format='pdf', bbox_inches='tight')
        plt.show()   
        
        t = 1.0* self.env.t
        
        return t, S, X, a

    def plot_policy(self, name=""):
        '''
        plot policy for various states combinations at different time instances from 0 to self.env.T

        '''
        
        NS = 51
        S = torch.linspace(0, 1.5 * self.env.pen, NS)
        
        NX = 51
        X = torch.linspace(-0.1, self.env.X_max, NX)
        
        Sm, Xm = torch.meshgrid(S, X,indexing='ij')

        def plot(k, lvls, title):
            
            # plot 
            fig, axs = plt.subplots(2, 2)
            plt.suptitle(title, y =1.01, fontsize = 'xx-large')
            
            t_steps = [0, self.env.T/4, self.env.T/2, (self.env.T - self.env.dt)]
            
            for idx, ax in enumerate(axs.flat):
                t = torch.ones(NS,NX) * t_steps[idx]
                Y = self.__stack_state__(t, Sm, Xm)
                
                # normalize : Y (tSX)
                a = self.pi['net'](self.normalize(Y, 'state').to(torch.float32)).detach().squeeze()
                cs = ax.contourf(Sm.numpy(), Xm.numpy(), a[:,:,k], 
                                  levels=lvls,
                                  cmap='RdBu')
                # print(torch.amin(a[:,:,0]),torch.amax(a[:,:,0]))
    
                ax.axvline(self.env.S0, linestyle='--', color='k')
                ax.axhline(self.env.R, linestyle='--', color='k')
                ax.set_title(r'$t={:.3f}'.format(t_steps[idx]) +'$',fontsize = 'x-large')
            
            fig.text(0.5, -0.01, 'OC Price', ha='center',fontsize = 'x-large')
            fig.text(-0.01, 0.5, 'Inventory', va='center', rotation='vertical',fontsize = 'x-large')
            # fig.subplots_adjust(right=0.9)   
    
            cbar_ax = fig.add_axes([1.04, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(cs, cax=cbar_ax)
            # cbar.set_ticks(np.linspace(-self.env.nu_max/2, self.env.nu_max/2, 11))
            # cbar.set_ticks(np.linspace(-50, 50, 11))
                
            plt.tight_layout()
            plt.show()
        
        plot(0, 
             np.linspace(-self.env.nu_max/2, self.env.nu_max/2, 21), 
             "Trade Rate Heatmap over Time")
        plot(1, 
             np.linspace(0,1,21),
             "Generation Probability Heatmap over Time")    
