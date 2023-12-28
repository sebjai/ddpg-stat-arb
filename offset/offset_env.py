# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 11:41:23 2022

@author: sebja
"""

import numpy as np
import tqdm
import pdb
import torch


class offset_env():

    def __init__(self, T=1/12, sigma=0.5, kappa=0.03, eta = 0.05, xi=0.1,
                     c=0.25, S0=2.5, R=5, pen=2.5, N=51):
        
        self.T=T
        self.sigma = sigma
        self.kappa = kappa
        self.eta = eta
        self.xi = xi
        self.c = c
        self.S0 = S0
        self.R = R 
        self.pen = pen
        self.X_max = 1.5 * R
        self.nu_max = 100.0
        
        self.N = N
        self.t = np.linspace(0,self.T, self.N)
        self.dt = self.t[1]-self.t[0]  # time steps
        self.inv_vol = self.sigma * np.sqrt(0.5*self.T )
        
    def randomize(self, mini_batch_size=10):
        # experiment with distributions
        # penalty + N(0,1)
        S0 = self.S0 + torch.randn(mini_batch_size) * self.inv_vol 
        # Unifrom(0,x_max)
        X0 = torch.rand(mini_batch_size) * self.X_max
        # randomized time 
        t0 = torch.tensor(np.random.choice(self.t[:-1], size=mini_batch_size, replace=True)).to(torch.float32)
        idx = np.random.choice(np.linspace(0, mini_batch_size-1), size=int(0.05 * mini_batch_size), replace=False)
        t0[idx] = (self.T - self.dt)
        
        return t0, S0, X0
      
    def step(self, y, a):
        
        mini_batch_size = y.shape[0]
        
        # G = 1 is a generate a credit by investing in a project
        G = 1 * (a[:,1] > torch.rand(mini_batch_size))
        
        yp = torch.zeros(y.shape)
        
        # time evolution
        yp[:,0] = y[:,0] + self.dt
        
        # asset price evolution
        yp[:,1] = y[:,1] \
            + (self.pen-y[:,1])/(self.T-y[:,0]) * self.dt \
                + self.sigma * np.sqrt(self.dt) * torch.randn(mini_batch_size) \
                    - self.eta * self.xi * G
                    
        # inventory evolution
        nu = (1-G) * a[:,0]
        yp[:,2] = y[:,2] + self.xi * G + nu * self.dt
        
        # Reward
        r = -( y[:,1] * nu *self.dt \
              + 0.5 * self.kappa * nu**2 * self.dt \
                  + self.c * G \
                      + self.pen * (torch.abs(yp[:,0]-self.T)<1e-8).int() * torch.maximum(self.R - yp[:,2], torch.tensor(0)))
        
        return yp, r
