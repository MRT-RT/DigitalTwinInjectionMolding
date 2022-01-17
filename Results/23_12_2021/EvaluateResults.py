# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 09:35:34 2022

@author: alexa
"""

import pickle as pkl
import pandas as pd

df = pd.DataFrame(columns=['loss_val'])

for i in range(1,14):
    c = pkl.load(open('LSTM_Durchmesser_innen_c'+str(i)+'.pkl','rb'))
    
    dfc = pd.DataFrame(c['loss_val'].min(),columns=['loss_val'],index=[i])
    df = df.append(dfc)
    
    