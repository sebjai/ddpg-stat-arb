# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:36:03 2024

@author: sebja
"""
#%%
import numpy as np
import torch
import matplotlib.pyplot as plt 
from gru_pred import gru_pred
import statsmodels.api as sm

#%%
model = gru_pred(T=100, 
                 learning_rate = 0.001,
                 seq_length=100, n_ahead=1, 
                 gru_hidden_size = 5, gru_num_layers = 5,
                 dropout_rate=0, kappa=5, sigma=0.3, dt=0.1, theta=[1., -1.])

#%%
model.train(num_epochs = 5_000, n_print=500)
# %%
torch.save(model.model.state_dict(), 'model.pth')
# %%
model.model.load_state_dict(torch.load('model.pth'))
model.pred()

#mod_hamilton = sm.tsa.MarkovAutoregression(
#    dta_hamilton, k_regimes=2, order=4, switching_ar=False
#)
# %%
model.env.sigma




# %%
