#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:02:48 2022

@author: alexander
"""

from pathlib import Path
import sys

path_dim = Path.cwd().parents[1]
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous.PreProcessing import LoadFeatureData, MinMaxScale
from DIM.models.model_structures import Static_Multi_MLP
from DIM.optim.param_optim import ParallelModelTraining, static_mode
from DIM.optim.param_optim import BestFitRate

import multiprocessing
import numpy as np
import pickle as pkl
import pandas as pd

from functions import Eval_MLP


# %% Chose data which to evaluate on an path to models


data = pkl.load(open('data_doubleExp.pkl','rb'))

data_train = data['data_train']
data_test = data['data_test']

results = []

folder = 'MLP_2layers_p0'

# %% Evaluate models on data

for h in range(1,11):
       
    opt_res = pkl.load(open(folder+'/QM_MLP_Di_h'+str(h)+'.pkl','rb'))
    
    model = opt_res['model']
    param_est = opt_res['est_params']
    
    if h == 1:
        minmax = opt_res['minmax']
  
        
    for init in param_est.index:    
        
        model.Parameters = param_est.loc[init,'params_val']
        
        results_train,results_val = Eval_MLP(model,data_train,data_test,minmax)
        
        pkl.dump({'results_train':results_train,'results_val':results_val},
                 open(folder+'/Pred_'+folder+'_h'+str(h)+'_init'+\
                      str(init)+'.pkl','wb'))
       
        BFR = results_val['BFR']
        
        print('dim c:'+str(h)+' init:' + str(init) + ' BFR: ' + 
              str(BFR))
        
        results.append([BFR,folder,h,'Durchmesser_innen',init])
        
df = pd.DataFrame(data=results,columns=['BFR','model','complexity','target','init'])

pkl.dump(df,open(folder+'/Results_'+folder+'.pkl','wb'))

# pkl.dump([results_train,results_val],open('GRU_4c_pred.pkl','wb'))