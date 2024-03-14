# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:36:03 2024

@author: sebja
"""

import numpy as np
import torch
import matplotlib.pyplot as plt 
from gru_pred import gru_pred


model = gru_pred(T=10, seq_length=50, n_ahead=10, dropout_rate=0)
model.train()
