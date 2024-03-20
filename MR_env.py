# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 11:41:23 2022

@author: sebja
"""

import numpy as np
import tqdm
import pdb
import torch
import scipy.linalg

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
        
        self.inv_vol = self.sigma/np.sqrt(2.0*self.kappa)
        self.eff_vol = self.sigma* np.sqrt((1-np.exp(-2*self.kappa*self.dt))/(2*self.kappa))
        
        self.I_max = I_max

        self.jump_intensity = 0.15
        self.time_arrival = 50#1/self.jump_intensity

    def fetch_theta(self, t, method = 'mc', t_exp=50):
        '''
        pmts: mc: markov chain, 
              poi: poisson, 
              exp: exponential -> jump at a fixed time t_exp

        '''
        theta = self.theta    

        if method == 'mc':
            if t == 0:
                theta = self.theta
            states = np.array([0.5, 1, 1.5])
            init_probs = np.array([0.3333333, 0.3333333, 0.3333333])
            trans_rate_matrix = np.array([[-0.0333333 ,  0.01666667,  0.01666667],
                                  [ 0.01666667, -0.0333333 ,  0.01666667],
                                  [ 0.01666667,  0.01666667, -0.0333333 ]])
            probabilities = init_probs + (init_probs * trans_rate_matrix * self.dt) # changethis
            theta = np.random.choice(states, p = np.round(probabilities[0], 5))

        elif method == 'poi':

            jump_count = np.random.poisson(self.jump_intensity * self.dt)#exponential(scale=50)#(self.jump_intensity * self.dt)

            if jump_count > 0:
                theta = np.random.choice([0.5, 1.5])
                  
        elif method == 'exp':

            if t == t_exp and t > 0:
                theta = np.random.choice([0.5, 1.5])

        return theta
        
    def lognormal(self, sigma, mini_batch_size=10):
        return torch.exp(-0.5*sigma**2 + sigma*torch.randn(mini_batch_size))
        
    def Randomize_Start(self, mini_batch_size=10):
        
        #S = self.S_0 + 3*self.inv_vol*torch.randn(mini_batch_size, self.N)
        #I = self.I_max * (2*torch.rand(mini_batch_size, self.N)-1)
        #self.S_0 + 3*self.inv_vol*torch.randn(mini_batch_size, )
        S, _ = self.Simulate(1,
                             0, mini_batch_size)
        I = self.I_max * (2*torch.rand(mini_batch_size, self.N)-1)
        return S, I

    def Simulate(self, s0, i0,  mini_batch_size=10):

        S = torch.zeros((mini_batch_size, self.N)).float()
        I = torch.zeros((mini_batch_size, self.N)).float()

        S[:, 0] = s0#self.S_0
        I[:, 0] = i0
        self.theta = 1 # reset the theta to the original value
        t_exp = np.round(np.random.exponential(50))

        for t in (range(self.N-1)):

            if t>0:
                self.theta = self.fetch_theta(t, 'exp', t_exp)

            S[:, t+1], I[:,t+1], _ = self.step(t*self.dt, S[:,t], I[:,t], 0*I[:,t], self.theta)
        
        return S, I
    
    def step(self, t, S, I, I_p, theta):
        '''
        Generates the next state of the environment 
        '''
        
        mini_batch_size = S.shape[0]
        
        S_p = theta + (S-theta)*np.exp(-self.kappa*self.dt) \
            + self.eff_vol *  torch.randn(S.shape)

        q = I_p-I

        r = I_p*(S_p-S) - self.lambd*torch.abs(q)

        return S_p, I_p, r
