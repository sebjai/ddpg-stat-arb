# -*- coding: utf-8 -*-
"""
Created on Mon May  13 13:39:56 2024

@author: macrandrea
"""
from MR_env_ddpg import MR_env as Environment
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
from datetime import datetime
import scipy.linalg as linalg 
import random
import pdb
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation, PillowWriter

class gru(nn.Module):
    def __init__(self, input_size, gru_hidden_size, gru_num_layers, output_size, lin_hidden_size=128, dropout_rate=0.0):
        super(gru, self).__init__()
        self.hidden_size = gru_hidden_size
        self.gru = nn.GRU(input_size, gru_hidden_size, num_layers=gru_num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(gru_hidden_size, lin_hidden_size)
        
        self.prop_h_to_h = nn.ModuleList([nn.Linear(lin_hidden_size, lin_hidden_size) for n in range(10)])
        
        self.fc3 = nn.Linear(lin_hidden_size, output_size)  # Output linear layer

    def forward(self, x):
        _, out = self.gru(x)  # out has shape (num_layers, batch, hidden_size)

        out = out[-1, :, :]  # Take the last hidden state from the GRU output (shape will be (batch, hidden_size))

        out = F.relu(self.fc1(out))
        
        for prop in self.prop_h_to_h:
            out = F.relu(prop(out))
        
        out = self.fc3(out)  # Now `out` has shape (batch, output_size)
        out = out.squeeze(-1)  # Squeeze the last dimension to ensure output shape is (batch,)

        return (out)

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

    def __init__(self, env: Environment, gru: None , I_max=10, 
                 gamma=0.99,  
                 n_nodes=36, n_layers=6, 
                 seq_length = 10,
                 n_ahead    = 1,
                 lr=1e-3, sched_step_size = 50, tau=0.001,
                 name=""):

        self.gamma = gamma
        self.I_max = I_max
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.name =  name
        self.sched_step_size = sched_step_size
        self.lr = lr
        self.seq_length = seq_length
        self.n_ahead    = n_ahead   
        #self.lg = nn.Softmax(dim=1)
        self.tau = tau
        
        self.__initialize_NNs__()
        
        self.S = []
        self.I = []
        self.q = []
        self.r = []
        self.epsilon = []
        
        self.Q_loss = []
        self.gru_loss = []
        self.pi_loss = []
        self.env = env

    def __initialize_NNs__(self):
        # gru for stock price prediction 
        #
        # features = S
        #
        self.gru = {'net': gru(input_size=1,
                                gru_hidden_size=20,
                                gru_num_layers=10,
                                output_size=1,
                                lin_hidden_size=128,
                                dropout_rate=0.0)}  
                                 
        self.gru['optimizer'] = optim.AdamW(self.gru['net'].parameters(), lr=self.lr)
        self.gru['scheduler'] = optim.lr_scheduler.StepLR(self.gru['optimizer'], step_size=self.sched_step_size, gamma=0.99)

        # policy approximation
        #
        # features = S, I
        #
        self.pi = {'net': ANN(n_in=2, 
                              n_out=1, 
                              nNodes=self.n_nodes, 
                              nLayers=self.n_layers,
                              out_activation='tanh',
                              scale=self.I_max,
                              )}
        
        self.pi['optimizer'], self.pi['scheduler'] = self.__optim_and_scheduler__(self.pi)        
        
        # Q - function approximation
        #
        # features = S, I, a=Ip
        #
        self.Q_main = {'net' : ANN(n_in=3, 
                                  n_out=1,
                                  nNodes=self.n_nodes, 
                                  nLayers=self.n_layers) }

        self.Q_main['optimizer'], self.Q_main['scheduler'] = self.__optim_and_scheduler__(self.Q_main)
        self.Q_target = copy.deepcopy(self.Q_main)
        
    def __optim_and_scheduler__(self, net):
        
        optimizer = optim.AdamW(net['net'].parameters(),
                                lr=self.lr)
            
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.sched_step_size,  gamma=0.99)
    
        return optimizer, scheduler
    
    def __stack_state__(self, S, I):

        return torch.cat((
            (S/self.env.S_0-1.0).unsqueeze(-1), #S.unsqueeze(-1),#
            (I/self.I_max  ).unsqueeze(-1), 
            ), axis=-1)
  
    def __grab_mini_batch__(self, mod_type = 'MC', mini_batch_size = 256):
        
        t = torch.rand((mini_batch_size))*self.env.N
        S, I, theta_true = self.env.Randomize_Start(type_mod = mod_type, batch_size = mini_batch_size)

        return t, S, I
    
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
            
        S, _, theta_true, kappa = self.env.Simulate(s0=self.env.S_0 + 3*self.env.start_inv_vol*torch.randn(mini_batch_size, ), 
                                             i0=self.I_max * (2*torch.rand(mini_batch_size)-1), 
                                             model = 'MC', batch_size=mini_batch_size, ret_reward = False, 
                                             I_p = 0, N = N)

        I = self.I_max * (2*torch.rand(mini_batch_size, S.shape[1])-1)
        
        if train == True:   

            return S, I
        
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

            batch_S, batch_I = self.obtain_data(mini_batch_size, N = 15)

            self.gru['net'].zero_grad()

            S_gru, y = self.create_snippets(batch_S[:, : self.seq_length+1].T)

            S = self.gru['net'](S_gru.T/self.env.S_0-1.0)#

            self.Q_main['optimizer'].zero_grad()

            # concatenate states
            X = self.__stack_state__(
                                    S.detach().squeeze(-1), 
                                    batch_I[:, self.seq_length+1],
                                    )

            I_p = self.pi['net'](X).detach() * torch.exp(-0.5*epsilon**2+epsilon*torch.randn(batch_S.shape[0],1))

            Q = self.Q_main['net']( torch.cat((X, I_p/self.I_max),axis=-1) )

            # step in the environment
            r = self.make_reward(batch_S[:, self.seq_length+1], 
                                 batch_S[:, self.seq_length+self.n_ahead+1], 
                                 batch_I[:, self.seq_length+1], I_p)

            # compute the Q(S', a*)

            S_gru_p, y = self.create_snippets(batch_S[:, 1: self.seq_length+self.n_ahead+1].T)

            S_p = self.gru['net'](S_gru_p.T/self.env.S_0-1.0)#
            
            X_p = self.__stack_state__(S_p.detach().squeeze(-1),
                                        I_p.squeeze(-1))   

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

            #loss_gru = torch.mean((S.squeeze(-1) - y)**2)
            #self.gru['optimizer'].step()
            #self.gru['scheduler'].step()
            #self.gru_loss.append(loss_gru.item())   
            #loss_gru.backward()

            self.soft_update(self.Q_main['net'], self.Q_target['net'])
        
    def soft_update(self, main, target):
    
        for param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
    def Update_pi(self, mini_batch_size=256, epsilon=0.02, n_iter_pi=10):

        for i in range(n_iter_pi):
            
            batch_S, batch_I = self.obtain_data(mini_batch_size, N = 15)

            S_gru, y = self.create_snippets(batch_S[:, : self.seq_length+1].T)

            #self.gru['net'](S_[:, t: t+self.seq_length].unsqueeze(-1)).detach()

            S = self.gru['net'](S_gru.T/self.env.S_0-1.0).detach()

            self.pi['optimizer'].zero_grad()

            X = self.__stack_state__(S.squeeze(-1), 
                                     batch_I[:,self.seq_length+1],  
                                     ) 
        
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

        #S = torch.zeros((1, N+2)).float()         
        I = torch.zeros((1, N+12)).float()
        r = torch.zeros((1, N+2)).float()

        S_ = self.obtain_data(mini_batch_size   =  1, N = N, train = False)

        for t in range(N-self.seq_length-1):

            S = self.gru['net'](S_[:, t: t+self.seq_length].unsqueeze(-1)).detach()
            

            X = self.__stack_state__(S, I[:, t+self.seq_length].T)

            I[:, t+self.seq_length+1] = self.pi['net'](X).reshape(-1).detach().unsqueeze(-1)

            r[:, t+1] = self.make_reward(S_[:,t+self.seq_length], 
                                         S_[:,t+self.seq_length+1],
                                         I[:,t+self.seq_length], 
                                         I[:,t+self.seq_length+1])

        I  =    I.detach().numpy()
        r =     r.detach().numpy()


        if no_plots == False:
            a = self.env.dt*np.arange(0, N)/self.env.T
            t = a[:-(self.seq_length + 2)]
            plt.figure(figsize=(5,5))

            fig, ax1 = plt.subplots()

            ax1.plot((S_[:,self.seq_length:-2]).squeeze(0).numpy(), label = 'Stock price')
            ax1.set_ylabel('Stock price')
            ax1.set_xlabel('Time')
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            ax2.plot((I[:,self.seq_length:-2]).squeeze(0), color='red', label = 'Inventory')
            ax2.set_ylabel('Inventory')
            ax2.legend(loc='upper right')

            plt.title("Stock price and Inventory")
            plt.show()

            plt.plot(np.cumsum(r.squeeze(0)))
            plt.title("cumulative return")

            plt.tight_layout()
        
            plt.savefig("path_"  +self.name + "_" + name + ".pdf", format='pdf', bbox_inches='tight')
            plt.show()

            return r, S, I
        
        if no_plots == True:
            return r , S, I  

    def run_strategy_rolling(self, name=datetime.now().strftime("%H_%M_%S"), N=12, no_plots=False):
        S = torch.zeros((1, N+2)).float()         
        I = torch.zeros((1, N)).float()
        r = torch.zeros((1, N+2)).float()
        theta_post = torch.zeros((1, N+2)).float()
    
        S = self.obtain_data(mini_batch_size=1, N=N, train=False)
    
        for t in range(N-self.seq_length):
    
            X = self.__stack_state__(S[:, t+self.seq_length-1].T, I[:, t+self.seq_length-1].T)
    
            I[:, t+self.seq_length] = self.pi['net'](X).reshape(-1).detach().unsqueeze(-1)
    
            r[:, t+1] = self.make_reward(S[:,t+self.seq_length-1], 
                                         S[:,t+self.seq_length],
                                         I[:,t+self.seq_length-1], 
                                         I[:,t+self.seq_length])
    
        I = I.detach().numpy()
        r = r.detach().numpy()
    
        if not no_plots:
            a = self.env.dt * np.arange(0, N) / self.env.T
            time_step = 100  # Length of each time frame
            t = np.arange(0, N-(self.seq_length + 2))
            max_t = t[-1]
            for start_t in range(int(max_t) - time_step + 1):  # Iterate one time step at a time
                end_t = start_t + time_step
                if end_t > max_t:
                    break
                
                t_snapshot = t[start_t:end_t]
                S_snapshot = S[:, self.seq_length + start_t:self.seq_length + end_t].squeeze(0).numpy()
                I_snapshot = I[:, self.seq_length + start_t:self.seq_length + end_t].squeeze(0)
    
                plt.figure(figsize=(5, 5))
                fig, ax1 = plt.subplots()
    
                ax1.plot(t_snapshot, S_snapshot, label='Stock price')
                ax1.set_ylabel('Stock price')
                ax1.set_ylim(self.env.S_0 - 5 * self.env.inv_vol, self.env.S_0 + 5 * self.env.inv_vol)
                ax1.set_xlabel('Time')
                ax1.legend(loc='upper left')
    
                ax2 = ax1.twinx()
                ax2.plot(t_snapshot, I_snapshot, color='red', label='Inventory')
                ax2.set_ylim(-self.I_max / 2, self.I_max / 2)         
                ax2.set_ylabel('Inventory')
                ax2.legend(loc='upper right')
    
                plt.title(f"Stock price and Inventory - t={start_t} to t={end_t}")
                plt.savefig(f"path_{int(start_t)}.png", format='png', bbox_inches='tight')
                plt.show()
    
        return r, S, I
        
    def run_strategy_rolling_gif(self, name=datetime.now().strftime("%H_%M_%S"), N=12, no_plots=False):
        S = torch.zeros((1, N+2)).float()         
        I = torch.zeros((1, N+2)).float()
        r = torch.zeros((1, N+2)).float()
        theta_post = torch.zeros((1, N+2)).float()
    
        S = self.obtain_data(mini_batch_size=1, N=N, train=False)
    
        for t in range(N-self.seq_length):
            theta_post = self.get_theta(S[:, t:t+self.seq_length+1])
    
            X = self.__stack_state__(S[:, t+self.seq_length-1].T, I[:, t+self.seq_length-1].T, theta_post)
    
            I[:, t+self.seq_length] = self.pi['net'](X).reshape(-1).detach().unsqueeze(-1)
    
            r[:, t+1] = self.make_reward(S[:,t+self.seq_length-1], 
                                         S[:,t+self.seq_length],
                                         I[:,t+self.seq_length-1], 
                                         I[:,t+self.seq_length])
    
        I = I.detach().numpy()
        r = r.detach().numpy()
    
        if no_plots == False:
            a = self.env.dt * np.arange(0, N) / self.env.T
            time_step = 100  # Length of each time frame
            t = np.arange(0, N-(self.seq_length + 2))#a[:-(self.seq_length + 2)]  
            max_t = t[-1]
            
            fig, ax1 = plt.subplots(figsize=(5, 5))
            ax2 = ax1.twinx()
            lines = [ax1.plot([], [], label='Stock price')[0], ax2.plot([], [], color='red', label='Inventory')[0]]
            
            def init():
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Stock price')
                ax1.set_ylim(self.env.S_0 - 5 * self.env.inv_vol, self.env.S_0 + 5 * self.env.inv_vol)
                ax1.legend(loc='upper left')
                ax2.set_ylim(-self.I_max/2, self.I_max/2)
                ax2.set_ylabel('Inventory')
                ax2.legend(loc='upper right')
                return lines
            
            def update(frame):
                start_t = frame
                end_t = min(start_t + time_step, max_t)
    
                t_snapshot = t[start_t:end_t]
                S_snapshot = S[:, self.seq_length + start_t:self.seq_length + end_t].squeeze(0).numpy()
                I_snapshot = I[:, self.seq_length + start_t:self.seq_length + end_t].squeeze(0)
    
                lines[0].set_data(t_snapshot, S_snapshot)
                lines[1].set_data(t_snapshot, I_snapshot)
                ax1.set_xlim(t_snapshot[0], t_snapshot[-1])
                plt.title(f"Stock price and Inventory - t={start_t} to t={end_t}")
                return lines

            ani = FuncAnimation(fig, update, frames=range(0, round(max_t), time_step), init_func=init, blit=True)
            
            video_path = f'strategy_animation_{name}.gif'  # Save as GIF
            writer = PillowWriter(fps=2)
            ani.save(video_path, writer=writer)
            plt.close(fig)
            return r, S, I
        
        if no_plots == True:
            return r
     
    def plot_policy(self, name=""):
        NS = 101
        S = torch.linspace(self.env.S_0 - 5*self.env.inv_vol,
                           self.env.S_0 + 5*self.env.inv_vol,
                           NS)
        NI = 101
        I = torch.linspace(-self.I_max/2, self.I_max/2, NI)
        Sm, Im = torch.meshgrid(S, I, indexing='ij')

        def plot(a, ax):
            cs = ax.contourf(Sm.squeeze().numpy(), Im.squeeze().numpy(), a.numpy(),
                             levels=np.linspace(-self.I_max/2, self.I_max/2, 21),
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
        pi_all = [(0.8, 0.1, 0.1),  # pi1 high
                  (0.1, 0.8, 0.1),  # pi2 high
                  (0.1, 0.1, 0.8)]  # pi3 high

        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        for i in range(len(pi_all)):
            pi1, pi2, pi3 = pi_all[i]
            #theta_estim = torch.cat((pi1 * ones, pi2 * ones, pi3 * ones), axis=-1)
            X = torch.cat(((Sm / self.env.S_0 - 1.0),
                           (Im / (self.I_max/2)),
                           ), axis=-1)
            a = self.pi['net'](X).detach().squeeze()
            cs = plot(a, ax[i])
            #ax[i].set_title(f"$\pi_{{{i+1}}}$ high")
            ax[0].set_title(f"$\phi_1$ high \n $\\theta = 0.9$")
            ax[1].set_title(f"$\phi_2$ high \n $\\theta = 1$")
            ax[2].set_title(f"$\phi_3$ high \n $\\theta = 1.1$")

        # Create an axis for the colorbar on the right side of the figure
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(cs, cax=cbar_ax)
        cbar.set_ticks(np.linspace(-self.I_max/2, self.I_max/2, 11))
        cbar.ax.set_ylabel('Action')

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Price")
        plt.ylabel("Inventory")
        plt.show()
