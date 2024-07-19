# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 11:41:23 2022

@author: sebja
"""

import numpy as np
import tqdm
import pdb
import torch
import scipy.linalg as linalg 
import random

class MR_env():

    def __init__(self, S_0=1300,
                 theta=1300,
                 kappa=5,
                 sigma=1,
                 dt = 1,
                 T=int(60*60),
                 I_max = 10,
                 lambd = 0.02):
        
        self.S_0 = S_0
        self.theta = theta
        self.sigma = sigma
        self.kappa = kappa
        self.lambd = lambd
        
        self.dt = dt  # time steps
        self.T = T
        self.N = int(self.T/self.dt)+1
        self.t = np.linspace(0,self.T, self.N)
        
        self.inv_vol = self.sigma/np.sqrt(2.0*self.kappa)
        self.eff_vol = self.sigma* np.sqrt((1-np.exp(-2*self.kappa*self.dt))/(2*self.kappa))
        
        self.I_max = I_max
        
    def lognormal(self, sigma, batch_size=10):
        return torch.exp(-0.5*sigma**2 + sigma*torch.randn(batch_size))
        
    def Randomize_Start(self, type_mod = 'MC', batch_size=10):
        
        #S = self.S_0 + 3*self.inv_vol*torch.randn(batch_size, self.N)
        #I = self.I_max * (2*torch.rand(batch_size, self.N)-1)
        S, _, theta = self.Simulate(self.S_0 + 3*self.inv_vol*torch.randn(batch_size, ),
                             self.I_max * (2*torch.rand(batch_size, self.N)-1), model = type_mod, batch_size = batch_size)
        I = self.I_max * (2*torch.rand(batch_size, self.N)-1)
        
        return S, I, theta

    def Simulate(self, s0, i0, model = 'exp', batch_size=10, ret_reward = False, I_p = 0, N = None):
        if N is None:
            N = self.N
        else :
            N = N
            
        S = torch.zeros((batch_size, N)).float()
        I = torch.zeros((batch_size, N)).float()
        I_p = torch.zeros(N).float()
        theta = torch.zeros((batch_size, N)).float()
        r = torch.zeros((batch_size, N)).float()
        Z = torch.zeros((batch_size, N)).int()
        theta[:] = torch.nan
        
        theta[:,0] = self.theta[0]
        
        S[:, 0] = s0 #self.S_0
        I[:, 0] = i0
        I_p[:] = I_p

        if model == 'exp':
            tau = np.sort(-np.log(np.random.rand(2))/0.2)

            for t in (range(N-1)):

                if self.t[t] < tau[0]:

                    theta[:,t] = self.theta[0]

                elif self.t[t] < tau[1]:

                    theta[:,t] = self.theta[1]

                else:

                    theta[:,t] = self.theta[2]

                S[:, t+1], I[:,t+1], r[:, t] = self.step(t*self.dt, S[:,t], I[:,t], I_p[:,t], theta[:,t])

            if ret_reward == True:
                return S, I, theta, r
            else:
                return S, I, theta
    
        elif model == 'MC':
            
            labels = torch.tensor(np.arange(3)).int()
            states = torch.tensor([0.9000, 1.0000, 1.1000])

            trans_rate_matrix = torch.tensor([[-0.1,  0.05,  0.05],
                                              [ 0.05, -0.1 ,  0.05],
                                              [ 0.05,  0.05, -0.1]])
            
            # probabilities = linalg.expm(trans_rate_matrix * self.dt)

            # theta[:,0] = torch.tensor(np.random.choice(states.numpy()))

            # for t in range(1, self.N-1):
                
            #     for j in range(batch_size):
            #         theta[j,t] = torch.tensor(np.random.choice(states, p = np.round(probabilities[np.where(states == theta[j, t-1])][0], 5)))#states[torch.choose(torch.tensor(probabilities), batch_size)]

            #     S[:, t+1], I[:,t+1], _ = self.step(t*self.dt, S[:,t], I[:,t], 0*I[:,t], theta[:,t])

            probs = torch.tensor(linalg.expm(trans_rate_matrix * self.dt)).float()
            cumsum_probs = torch.cumsum(probs, axis=1)

            Z[:,0] = torch.tensor(np.random.choice(np.arange(len(labels)), batch_size)).int()
            theta[:,0] = states[Z[:,0]] 

            for t in range(N-1):
                
                # step in the environment
                S[:, t+1], I[:,t+1], _ = self.step(t*self.dt, S[:,t], I[:,t], I[:, t+1], theta[:,t], batch_size = batch_size)
                
                # evolve the MC
                U = torch.rand(batch_size,1)
                Z[:,t+1] = torch.sum((cumsum_probs[Z[:,t]] < U), axis=1).int()
                theta[:,t+1] = states[Z[:,t+1]]

            if ret_reward == True:
                return S, I, theta, r
            else:
                return S, I, theta

    def step(self, t, S, I, I_p, theta, batch_size=10):
        
        #batch_size = S.shape[0]
        
        S_p = theta + (S-theta)*np.exp(-self.kappa*self.dt) \
            + self.eff_vol *  torch.randn(S.shape )#(S.shape)

        q = I_p-I

        r = I_p*(S_p-S) - self.lambd*torch.abs(q)

        return S_p, I_p, r
