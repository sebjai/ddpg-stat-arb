# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:38:51 2024

@author: sebja
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from MR_env_ddpg import MR_env
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import dill

class gru(nn.Module):
    def __init__(self, input_size, gru_hidden_size, gru_num_layers, output_size, 
                 lin_hidden_size=128, 
                 dropout_rate=0.0):
        
        super(gru, self).__init__()
        
        self.hidden_size = gru_hidden_size
        self.gru = nn.GRU(input_size, gru_hidden_size, num_layers=gru_num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(gru_hidden_size*gru_num_layers, lin_hidden_size)
        
        self.prop_h_to_h = nn.ModuleList([nn.Linear(lin_hidden_size, lin_hidden_size) for n in range(5)])
        
        self.fc3 = nn.Linear(lin_hidden_size, output_size)  # Output linear layer

    def forward(self, x):
        _, out = self.gru(x)

        out = F.silu(self.fc1(out.transpose(0,1).flatten(1,2)))
        
        for prop in self.prop_h_to_h:
            out = F.silu( prop(out))

        out = self.fc3(out)#F.softmax(, dim=1)
        
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
                 gru_hidden_size = 5,  # Number of units in the hidden state of the GRU layer
                 gru_num_layers = 5,  # Number of layers in the GRU layer 
                 dropout_rate = 0.7, # dropout rate   
                 I_max = 5,
                 n_ahead=1,
                 decay_rate=0.99,
                 seed = 10002197,
                 kappa=[1, 5, 10],
                 sigma=[0.3, 0.1],
                 theta=[0.9, 1, 1.1]): # torch.tensor([0.8000, 1.0000, 1.2000])

        np.random.seed(seed)  
        torch.manual_seed(seed)
        
        self.env = MR_env(S_0=1, kappa=kappa, sigma=sigma, theta=theta, dt=dt, T = T, I_max=I_max, lambd=0.05)
        
        self.seq_length = seq_length
        self.n_ahead = n_ahead
        
        #self.find_normalization()
        
        self.early_stopping = early_stopping(patience=500)
        
        self.model = gru( input_size=input_size, gru_hidden_size=gru_hidden_size,
                         gru_num_layers=gru_num_layers, 
                         output_size=len(theta), 
                         lin_hidden_size=64, dropout_rate = dropout_rate)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = decay_rate)
        
        self.losses = []      
        
    def find_normalization(self, batch_size=1_000):
        
        S, _, _ = self.env.Randomize_Start('MC', batch_size)
        
        self.a = torch.mean(S)
        self.b = torch.std(S)
        
    def grab_data(self, batch_size):
        
        S, _, theta, kappa, sigma = self.env.Randomize_Start('MC', batch_size)
        x_norm = S #(S - self.a)/self.b
        
        x, y = self.create_snippets(x_norm, theta)
        
        xx,k = self.create_snippets(x_norm, kappa)

        zz,s = self.create_snippets(x_norm, sigma)

        return x, y, k, s
    
    def create_snippets(self, x, theta=None):
        
        x_cp = x.T
        if theta is not None:
            theta_cp = theta.T
        
        Z = torch.zeros((self.seq_length, 
                         x_cp.shape[0]-self.seq_length-self.n_ahead, 
                         x_cp.shape[1]))
        
        Y = torch.zeros((x_cp.shape[0]-self.seq_length-self.n_ahead, 
                         x_cp.shape[1]))        
        
        for i in range(x_cp.shape[0]-self.seq_length-self.n_ahead):
            Z[:,i,:] = x_cp[i:i+self.seq_length,:]
            # Y[i,:] = x_cp[i+self.seq_length+(self.n_ahead-1),:] \
            #     - x_cp[i+self.seq_length+(self.n_ahead-1)-1,:]
            if theta is None:
                Y[i,:] = x_cp[i+self.seq_length+(self.n_ahead-1),:]
            else:
                Y[i,:] = theta_cp[i+self.seq_length,:]
        
        return Z.transpose(0,1), Y
    
    def handle_y(self, y):
            
        y_i = torch.zeros(y.shape[0], 3)
    
        for i in range(y.shape[0]):
    
            if y[i] == self.env.theta[0]:
    
                y_i[i,0] = 1
    
            elif y[i] == self.env.theta[1]:
    
                y_i[i,1] = 1
    
            elif y[i] == self.env.theta[2]:
    
                y_i[i,2] = 1
                
        return y_i

    def calc_accuracy(self, y, y_pred):

        y_i = self.handle_y(y)

        predicted_labels = torch.argmax(y_pred, dim=1)

        predictions = torch.zeros(len(self.env.theta)).repeat(y.shape[0], 1)

        for i in range(len(self.env.theta)):
        
            predictions[:, i] = (predicted_labels == y_i[:, i])

        return accuracy_score(y_i, predictions.numpy()) #classification_report(y_i, correct_predictions.numpy(), zero_division = 0.0)
    
    def plot_losses(self):
        
        losses = np.array(self.losses)
        l, l_err = self.moving_average(losses, 200)
        
        plt.plot(np.arange(len(l)), losses, alpha=0.1, color='k')
        plt.fill_between(np.arange(len(l)), l-l_err, l+l_err, alpha=0.2)
        plt.plot(np.arange(len(l)), l)
        plt.yscale('log')
        # plt.ylim(0.03,0.1)
        plt.show()
        
    def moving_average(self, x, n):
        
        y = np.zeros(len(x))
        y_err = np.zeros(len(x))
        y[0] = np.nan
        y_err[0] = np.nan        
        
        m1 = 0
        m2 = 0
        
        for i in range(1,len(x)):
            
            if i < n+1:
                
                m1 = np.mean(x[:i])
                m2 = np.mean(x[:i]**2)
                y[i] = m1
                y_err[i] = np.sqrt(m2 - m1**2)
                
            else:
                
                m1 = m1 + (1/n)*(x[i-1]-x[i-n-1])
                m2 = m2 + (1/n)*(x[i-1]**2-x[i-n-1]**2)
                
                y[i] = m1
                y_err[i] = np.sqrt(m2-m1**2)
                
        return y, y_err        
        
    def train(self, num_epochs=10_000, n_print=100):

        for epoch in tqdm(range(num_epochs)):
            
            # Forward pass
            x, y, k, s = self.grab_data(1)

            outputs = self.model(x)
            ell = nn.CrossEntropyLoss()
            loss =  ell(outputs, self.handle_y(y))

            #loss = 0
#
            #for k in range(len(self.env.theta)):
            #    mask = (y == self.env.theta[k]).squeeze()
            #    loss += -torch.sum(torch.log(outputs[mask,k]))
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.losses.append(np.sqrt(loss.item()))
            
            if (epoch+1) % 25 ==0:
                self.scheduler.step()
    
            if epoch == 0 or (epoch+1) % n_print == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')          
                
                accuracy = self.calc_accuracy(y, outputs)
                #print('Accuracy',accuracy)

                self.pred()
                self.plot_losses()
        #dill.dump(self.model, open('pred_model.pk','wb'))
                
                
                
                # benchmark con metodi più semplici specialmente in regimi con molto rumore
                # mettere più rumore sigma più grande della distanza fra i regimi
                # confrontare con la media mobile come naive 
                # confrontare con hidden markov model semplice
                # soluzione analitica? libro di ecnometria ar 1 regime switching    
                # markov switching autoregressive model - tsay
                # hamilton capitolo 22 - 1994
                # bayesian online learning con yvonni -> numero non finito di regimi -> modello classico di machine learning con dati iid e adattato al caso ar 1 
                # markoviano di ordine arbitrario

    def pred(self):
        
        x, y, kappa, sigma = self.grab_data(1)
        t = self.env.t[:-self.n_ahead-self.seq_length]
        
        lg = nn.Softmax(dim=1)
        pre = self.model(x).detach().squeeze()#.numpy()
        #logsoftmax = nn.LogSoftmax(dim=1)
        pred = lg( pre).numpy()
        fig  = plt.figure()
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        
        ax.plot(t, x[:,-1,0], label=r'$S_t$', color='k')
        for k in range(len(self.env.theta)):
            ax2.plot(t, pred[:,k], label=r"$\widehat{\mathbb{P}}[\theta_{t+n} = \theta^{(" + str(k) + ")}|\mathcal{F}_t]$")
            
        ax.plot(t, y, label=r'$\theta_{t}$', color='tab:red')#+n
        
        plt.xlabel(r"$t$")
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        ax2.legend(loc='upper left', bbox_to_anchor=(-0.43, 1))
        ax.set_ylabel(r'$S_t$')
        ax2.set_ylabel(r'$\widehat{\mathbb{P}}[\theta_{t+n} = \theta^{(k)}|\mathcal{F}_t]$')
        fig.text(0.5, 0.01, r'The possible levels for $\theta$ are: $\theta^{(0)}$ = 0.9, $\theta^{(1)}$ = 1, $\theta^{(2)}$ = 1.1 ', ha='center')
        np.save('S.npy', x)
        np.save('Y.npy', y)
        np.save('K.npy', kappa)
        
        plt.show()

        ###########################################################################
        fig  = plt.figure()
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        
        ax.plot(t, x[:,-1,0], label=r'$S_t$', color='k')

        ax2.plot(t, kappa, label=r"$\kappa_{t}$", color='tab:red')#+n
            
        
        
        plt.xlabel(r"$t$")
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        ax2.legend(loc='upper left', bbox_to_anchor=(-0.43, 1))
        ax.set_ylabel(r'$S_t$')
        ax2.set_ylabel(r'$\kappa_{t}$')
        fig.text(0.5, 0.01, r'The possible levels for $\kappa$ are: $\kappa^{(0)}$ = 3, $\kappa^{(1)}$ = 7', ha='center')
        
        plt.show()

        ###########################################################################
        fig  = plt.figure()
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        
        ax.plot(t, x[:,-1,0], label=r'$S_t$', color='k')

        ax2.plot(t, sigma, label=r"$\sigma_{t}$", color='tab:red')#+n
            
        
        
        plt.xlabel(r"$t$")
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        ax2.legend(loc='upper left', bbox_to_anchor=(-0.43, 1))
        ax.set_ylabel(r'$S_t$')
        ax2.set_ylabel(r'$\sigma_{t}$')
        fig.text(0.5, 0.01, r'The possible levels for $\sigma$ are: $\sigma^{(0)}$ = 0.1, $\sigma^{(1)}$ = 0.3', ha='center')
        
        plt.show()
