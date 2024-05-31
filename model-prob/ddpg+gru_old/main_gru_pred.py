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


model = gru_pred(T=21.1, 
                 learning_rate = 0.001,
                 seq_length=10, n_ahead=1, 
                 dropout_rate=0, kappa=5, sigma=0.1, batch_size=1)

model.train(num_epochs = 10_000, n_print=500)
# %%
torch.save(model.model.state_dict(), 'model.pth')
# %%
model.model.load_state_dict(torch.load('model.pth'))
model.pred()
# %%
