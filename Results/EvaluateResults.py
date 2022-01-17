# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 09:35:34 2022

@author: alexa
"""

import pickle as pkl
import pandas as pd

df = pd.DataFrame(columns=['LSTM 2 Inputs','LSTM 5 Inputs','GRU 2 Inputs',
                           'GRU 5 Inputs'], index = range(1,14))

path = '23_12_2021/'
for i in range(1,14):
    c = pkl.load(open(path+'LSTM_Durchmesser_innen_c'+str(i)+'.pkl','rb'))
    df.loc[i]['LSTM 2 Inputs'] = c['loss_val'].min()


path = '29_12_2021/'
for i in range(1,14):
    c = pkl.load(open(path+'LSTM_Durchmesser_innen_c'+str(i)+'.pkl','rb'))
    df.loc[i]['LSTM 5 Inputs'] = c['loss_val'].min()
        
path = '04_01_2022/'
for i in range(1,14):
    c = pkl.load(open(path+'GRU_Durchmesser_innen_c'+str(i)+'.pkl','rb'))
    df.loc[i]['GRU 5 Inputs'] = c['loss_val'].min()
    
path = '17_01_2022/'
for i in range(1,14):
    c = pkl.load(open(path+'GRU_Durchmesser_innen_c'+str(i)+'.pkl','rb'))
    df.loc[i]['GRU 2 Inputs'] = c['loss_val'].min()    