# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:16:22 2022

@author: LocalAdmin
"""

import sys
from pathlib import Path

path_dim = Path.cwd().parents[1]
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous.PreProcessing import LoadFeatureData, MinMaxScale
from DIM.models.model_structures import Static_Multi_MLP
from DIM.optim.param_optim import ModelTraining
from DIM.optim.control_optim import StaticProcessOptimizer

import multiprocessing

import pickle as pkl

res = pkl.load(open('MLP_2layers_T0/QM_MLP_Di_h10.pkl','rb'))
model = res['model']


data = pkl.load(open('data_doubleExp.pkl','rb'))

data_train = data['data_train']
data_test = data['data_test']

data_train,minmax = MinMaxScale(data_train,columns=model.u_label+model.y_label)
data_test,_ = MinMaxScale(data_test,minmax=minmax)

loss, pred = model.static_mode(data_test)

pred_un = MinMaxScale(pred,minmax=minmax,reverse=True)

test = StaticProcessOptimizer(model=model,fix_inputs=['T_wkz_0'])
test.optimize(1,1)

# if __name__ == '__main__':
    
#     multiprocessing.freeze_support()
    
#     for i in range(10,11):
#         res = Fit_MLP(i,data_train,data_test) 
        
        # pkl.dump(res,open('MLP_2layers_static/QM_MLP_Di_h'+str(i)+'.pkl','wb'))
   










