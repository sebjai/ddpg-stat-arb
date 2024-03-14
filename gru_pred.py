# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:38:51 2024

@author: sebja
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from MR_env import MR_env
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb

class gru(nn.Module):
    def __init__(self, input_size, gru_hidden_size, gru_num_layers, output_size, 
                 lin_hidden_size=128, 
                 dropout_rate=0.0):
        super(gru, self).__init__()
        self.hidden_size = gru_hidden_size
        self.gru = nn.GRU(input_size, gru_hidden_size, num_layers=gru_num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(gru_hidden_size*gru_num_layers, lin_hidden_size)
        
        self.prop_h_to_h = nn.ModuleList([nn.Linear(lin_hidden_size, lin_hidden_size) for n in range(3)])
        
        
        #self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second additional linear layer
        self.fc3 = nn.Linear(lin_hidden_size, output_size)  # Output linear layer

    def forward(self, x):
        _, out = self.gru(x)
        # out = self.dropout(out)
        out = F.silu(self.fc1(out.transpose(0,1).flatten(1,2)))
        for prop in self.prop_h_to_h:
            out = F.silu( prop(out))
        #out = F.leaky_relu(self.fc2(out))  # Only take the output of the last time step
        out = self.fc3(out)
        
        return out
    
class early_stopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'Early stopping after {self.counter} epochs of no improvement.')
                return True  # Stop training
        return False  # Continue training    

class gru_pred():
    
    def __init__(self, learning_rate = 0.001,
                 T = int(10),
                 dt = 0.1, 
                 seq_length=10,
                 input_size = 1,  # Number of features in each time step of the input sequence
                 gru_hidden_size = 30,  # Number of units in the hidden state of the GRU layer
                 gru_num_layers = 5,  # Number of layers in the GRU layer 
                 dropout_rate = 0.7, # dropout rate   
                 I_max = 5,
                 n_ahead=1,
                 decay_rate=0.99,
                 seed = 10002197):

        np.random.seed(seed)  
        torch.manual_seed(seed)
        
        self.env = MR_env(S_0=1, kappa=1, sigma=0.0001, theta=1, dt=dt, T = T, I_max=I_max, lambd=0.05)
        
        self.seq_length = seq_length
        self.n_ahead = n_ahead
        
        self.find_normalization()
        
        self.early_stopping = early_stopping(patience=500)
        
        self.model = gru( input_size=input_size, gru_hidden_size=gru_hidden_size,
                         gru_num_layers=gru_num_layers, 
                         output_size=1, 
                         lin_hidden_size=32, dropout_rate = dropout_rate)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = decay_rate)
        
        self.losses = []
        
        
    def find_normalization(self, batch_size=1_000):
        
        S, _ = self.env.Randomize_Start(batch_size)
        
        self.a = torch.min(S)
        self.b = torch.max(S) - self.a
        
        
    def grab_data(self, batch_size):
        
        S, _ = self.env.Randomize_Start(batch_size)
        x_norm = (2*(S - self.a)/self.b)-1
        
        x, y = self.create_snippets(x_norm)
        
        return x, y
    
    def create_snippets(self, x):
        
        x_cp = x.T
        
        Z = torch.zeros((self.seq_length, 
                         x_cp.shape[0]-self.seq_length-self.n_ahead, 
                         x_cp.shape[1]))
        
        Y = torch.zeros((x_cp.shape[0]-self.seq_length-self.n_ahead, 
                         x_cp.shape[1]))        
        
        for i in range(x_cp.shape[0]-self.seq_length-self.n_ahead):
            Z[:,i,:] = x_cp[i:i+self.seq_length,:]
            Y[i,:] = x_cp[i+self.seq_length+(self.n_ahead-1),:]
        
        return Z.transpose(0,1), Y
    
    
    def train(self, num_epochs=10_000):
        
        for epoch in tqdm(range(num_epochs)):
            
            # Forward pass
            x, y = self.grab_data(1)
            
            outputs = self.model(x)
            
            loss = torch.mean((y - outputs)**2)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.losses.append(loss.item())
            
            if (epoch+1) % 50 ==0:
                self.scheduler.step()
    
            if epoch == 0 or (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                
            # if epoch > 100:
            #     if early_stopping(loss.item()):
            #         print("stopping early")
            #         break   
    
        plt.plot(self.losses)
        plt.yscale('log')
        
    def pred(self):
        
        x, y = self.grab_data(1)
        
        pred = self.model(x).detach()
        
        plt.scatter(np.arange(y.shape[0]), y.squeeze().numpy(),s=10)
        plt.scatter(np.arange(y.shape[0]), pred.squeeze().numpy(),s=10)
        plt.show()
        
        plt.hist((y-pred).squeeze().numpy(), bins=np.linspace(-0.2,0.2,21), alpha=0.5)
        plt.hist((y-x[:,-1,:]).squeeze().numpy(), bins=np.linspace(-0.2,0.2,21), alpha=0.5)
        plt.show()
        
        model_err = (y-pred).squeeze().numpy()
        print( np.mean(model_err), np.std(model_err) )
        
        naive_err=(y-x[:,-1,:]).squeeze().numpy()
        print( np.mean(naive_err), np.std(naive_err) )
        
        S, _ = self.env.Randomize_Start(1)
        
        plt.plot(S.numpy().T)
        plt.show()
        
        
