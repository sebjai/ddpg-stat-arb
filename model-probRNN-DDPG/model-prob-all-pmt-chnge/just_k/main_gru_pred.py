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
import dill
import statsmodels.api as sm

#%%
model = gru_pred(T=40, 
                 learning_rate = 0.001,
                 seq_length=10, n_ahead=1, 
                 dropout_rate=0, kappa=[3, 7], theta=[0.9, 1, 1.1], sigma=0.3, dt=0.2)
#%%
model.train(num_epochs = 10_000, n_print=500)
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
