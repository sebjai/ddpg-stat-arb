# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:36:03 2024

@author: sebja
"""

import numpy as np
import torch
import matplotlib.pyplot as plt 
from gru_pred import gru_pred


model = gru_pred(T=20, 
                 learning_rate = 0.001,
                 seq_length=2, n_ahead=1, 
                 dropout_rate=0, kappa=5, sigma=0.1)
model.train(n_print=500)
