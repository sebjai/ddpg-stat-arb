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

#%%
model = gru_pred(T=20, 
                 learning_rate = 0.001,
                 seq_length=10, n_ahead=1, 
                 dropout_rate=0, kappa=5, sigma=0.1)
model.train(num_epochs = 30_000, n_print=500)

# %%
dill.dump(model, open('pred_model.pk','wb'))

#%%
model = dill.load(open('pred_model.pk','rb'))