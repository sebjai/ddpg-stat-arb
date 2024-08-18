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
        self.kappa = kappa
        self.sigma = sigma
        self.start_inv_vol = self.sigma[0]/np.sqrt(2.0*self.kappa[0])
        self.lambd = lambd
        
        self.dt = dt  # time steps
        self.T = T
        self.N = int(self.T/self.dt)+1
        self.t = np.linspace(0,self.T, self.N)
    
        self.I_max = I_max
        
    def lognormal(self, sigma, batch_size=10):
        return torch.exp(-0.5*sigma**2 + sigma*torch.randn(batch_size))
        
    def Randomize_Start(self, type_mod = 'MC', batch_size=10):
        
        #S = self.S_0 + 3*self.inv_vol*torch.randn(batch_size, self.N)
        #I = self.I_max * (2*torch.rand(batch_size, self.N)-1)
        #start_inv_vol = self.sigma/np.sqrt(2.0*self.kappa[0])
        S, _, theta, kappa, sigma = self.Simulate(self.S_0 + 3*self.start_inv_vol*torch.randn(batch_size, ),
                             self.I_max * (2*torch.rand(batch_size,)-1), model = type_mod, batch_size = batch_size)
        I = self.I_max * (2*torch.rand(batch_size, self.N)-1)
        
        #self.kappa = kappa[0]
        
        return S, I, theta, kappa, sigma

    def Simulate(self, s0, i0, model = 'exp', batch_size=10, ret_reward = False, I_p = 0, N = None):
        if N is None:
            N = self.N
        else :
            N = N
            
        S = torch.zeros((batch_size, N)).float()
        I = torch.zeros((batch_size, N)).float()
        I_p = torch.zeros(N).float()
        theta = torch.zeros((batch_size, N)).float()
        kappa = torch.zeros((batch_size, N)).float()
        sigma = torch.zeros((batch_size, N)).float()
        r = torch.zeros((batch_size, N)).float()
        Z = torch.zeros((batch_size, N)).int()
        Z_k = torch.zeros((batch_size, N)).int()
        Z_s = torch.zeros((batch_size, N)).int()
        theta[:] = torch.nan
        kappa[:] = torch.nan

        
        theta[:,0] = self.theta[0]
        kappa[:,0] = self.kappa[0]
        sigma[:,0] = self.sigma[0]
        
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
            #theta 
            labels = torch.tensor(np.arange(3)).int()
            states = torch.tensor([0.9000, 1.0000, 1.1000])

            trans_rate_matrix = torch.tensor([[-0.1,  0.05,  0.05],
                                              [ 0.05, -0.1 ,  0.05],
                                              [ 0.05,  0.05, -0.1]])

            probs = torch.tensor(linalg.expm(trans_rate_matrix * self.dt)).float()
            cumsum_probs = torch.cumsum(probs, axis=1)

            Z[:,0] = torch.tensor(np.random.choice(np.arange(len(labels)), batch_size)).int()
            theta[:,0] = states[Z[:,0]] 

            #kappa
            labels_k = torch.tensor(np.arange(2)).int()
            states_k = torch.tensor([3.0000, 7.0000])

            trans_rate_matrix_k = torch.tensor([[-0.1,  0.1],
                                               [ 0.1 , -0.1 ]])

            probs_k = torch.tensor(linalg.expm(trans_rate_matrix_k * self.dt)).float()
            cumsum_probs_k = torch.cumsum(probs_k, axis=1)

            Z_k[:,0] = torch.tensor(np.random.choice(np.arange(len(labels_k)), batch_size)).int()
            kappa[:,0] = states_k[Z_k[:,0]] 

            #sigma
            labels_s = torch.tensor(np.arange(2)).int()
            states_s = torch.tensor([0.3, 0.1])

            trans_rate_matrix_s = torch.tensor([[-0.1,  0.1],
                                               [ 0.1 , -0.1 ]])

            probs_s = torch.tensor(linalg.expm(trans_rate_matrix_s * self.dt)).float()
            cumsum_probs_s = torch.cumsum(probs_s, axis=1)

            Z_s[:,0] = torch.tensor(np.random.choice(np.arange(len(labels_s)), batch_size)).int()
            sigma[:,0] = states_s[Z_s[:,0]] 

            for t in range(N-1):
                
                # step in the environment
                S[:, t+1], I[:,t+1], _ = self.step(t*self.dt, S[:,t], I[:,t], I[:, t+1], theta[:,t], kappa[:,t], sigma[:, t], batch_size = batch_size)
                
                # evolve the MC
                U = torch.rand(batch_size,1)
                Z[:,t+1] = torch.sum((cumsum_probs[Z[:,t]] < U), axis=1).int()
                theta[:,t+1] = states[Z[:,t+1]]
                ####
                U_k = torch.rand(batch_size,1)
                Z_k[:,t+1] = torch.sum((cumsum_probs_k[Z_k[:,t]] < U_k), axis=1).int()
                kappa[:,t+1] = states_k[Z_k[:,t+1]]       
                ####
                U_s = torch.rand(batch_size,1)
                Z_s[:,t+1] = torch.sum((cumsum_probs_s[Z_s[:,t]] < U_s), axis=1).int()
                sigma[:,t+1] = states_s[Z_s[:,t+1]]                             

            if ret_reward == True:
                return S, I, theta, kappa, sigma, r
            else:
                return S, I, theta, kappa, sigma

    def step(self, t, S, I, I_p, theta, kappa, sigma, batch_size=10):
        
        inv_vol = sigma/np.sqrt(2.0*kappa)
        eff_vol = sigma* np.sqrt((1-np.exp(-2*kappa*self.dt))/(2*kappa))
        
        S_p = theta + (S-theta)*np.exp(-kappa*self.dt) \
            + eff_vol *  torch.randn(S.shape )

        q = I_p-I

        r = I_p*(S_p-S) - self.lambd*torch.abs(q)

        return S_p, I_p, r
