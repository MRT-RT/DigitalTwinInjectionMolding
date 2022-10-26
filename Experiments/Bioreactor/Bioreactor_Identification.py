# -*- coding: utf-8 -*-
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.io import loadmat
from pathlib import Path
import sys
from multiprocessing import freeze_support

dim_path = Path.cwd().parents[1]
sys.path.insert(0, dim_path.as_posix())

import DIM.models.lpv_models as lpv
from DIM.optim import param_optim



''' Data Preprocessing '''

# %% Load Data
train = loadmat('../Benchmarks/Bioreactor/APRBS_Data_3')
train = train['data']
val = loadmat('../Benchmarks/Bioreactor/APRBS_Data_1')
val = val['data']
test = loadmat('../Benchmarks/Bioreactor/APRBS_Data_2')
test = test['data']

# %%

################ Subsample Data ###############################################
train = train[0::50,:]
val = val[0::50,:]
test = test[0::50,:]
################# Pick Training- Validation- and Test-Data ####################

train = pd.DataFrame(data=train,columns=['u','y'])
val = pd.DataFrame(data=val,columns=['u','y'])
test = pd.DataFrame(data=test,columns=['u','y'])

init_state = np.zeros((2,1))

# %% 

data_train = {}
data_val = {}
data_test = {}

data_train['data'] = [train]
data_train['init_state'] = [init_state]

data_val['data'] = [val]
data_val['init_state'] = [init_state]

data_test['data'] = [test]
data_test['init_state'] = [init_state]

# %%

# Load inital linear state space model
LSS=loadmat("../Benchmarks/Bioreactor/Bioreactor_LSS")
LSS=LSS['Results']

initial_params = {'A0': LSS['A'][0][0],
                  'B0': LSS['B'][0][0],
                  'C0': LSS['C'][0][0]}


# %% Initialize instance of model

# model = lpv.LachhabLPV(dim_u=1,dim_x=2,dim_y=1,u_label=['u'],
#                                    y_label=['y'],dim_thetaA=1,dim_thetaB=1,
#                                    dim_thetaC=0,initial_params=initial_params,
#                                    name='lach1')

# model = lpv.RehmerLPV_old(dim_u=1,dim_x=2,dim_y=1,u_label=['u'], y_label=['y'], 
#                       name='ar',dim_thetaA=1,dim_thetaB=1,dim_thetaC=0,
#                       fA_dim=2,fB_dim=2,fC_dim=0,activation=0)

# model = lpv.RehmerLPV_outputSched(dim_u=1,dim_x=2,dim_y=1,u_label=['u'],
#                                   y_label=['y'],name='ar',dim_thetaA=1,
#                                   dim_thetaB=0,dim_thetaC=0,NN_1_dim=[5,1],
#                                   NN_2_dim=[],NN_3_dim=[],NN1_act=[1,0],
#                                   NN2_act=[],NN3_act=[])

# model = lpv.RehmerLPV(dim_u=1,dim_x=2,dim_y=1,u_label=['u'],
#                         y_label=['y'],name='ar',dim_thetaA=1,
#                         dim_thetaB=1,dim_thetaC=1,NN_1_dim=[5,1],
#                         NN_2_dim=[3,1],NN_3_dim=[2,1],NN1_act=[1,0],
#                         NN2_act=[1,0],NN3_act=[1,0])

# model = lpv.RBFLPV_outputSched(dim_u=1,dim_x=2,dim_y=1,u_label=['u'],
#                                y_label=['y'],name='ar',dim_theta=5,
#                                NN_dim=[],NN_act=[])

# model = lpv.RBFLPV(dim_u=1,dim_x=2,dim_y=1,u_label=['u'],
#                                y_label=['y'],name='ar',dim_theta=5,
#                                NN_dim=[],NN_act=[])

# model.InitializeLocalModels(initial_params['A0'],initial_params['B0'],
#                             initial_params['C0'])
 

model = lpv.Rehmer_NN_LPV(dim_u=1,dim_x=2,dim_y=1,u_label=['u'],y_label=['y'],
                          name='ar',dim_thetaA=1,NN_A_dim=[[4,4,1]],
                          NN_A_act=[[1,1,0]])
# %% Estimate parameters
if __name__ == '__main__':
    
    freeze_support()
    opti = param_optim.ParamOptimizer(model,data_train,data_val,
                                      res_path=Path.cwd()/'results2',
                                      n_pool=10)
    
    res = opti.optimize()
    
    idx = res['loss_val'].idxmin()
    
    model.parameters = res.loc[idx,'params_val']
    _,test_pred = model.parallel_mode(data_test)
    
    
    plt.close('all')
    plt.plot(data_test['data'][0]['y'])
    plt.plot(test_pred[0]['y'])
# %%

# ''' Call the Function ModelTraining, which takes the model and the data and 
# starts the optimization procedure 'initializations'-times. '''

# for dim in [1,2,3,4,5]:
    
#     model = NN.LachhabLPV_outputSched(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=dim,dim_thetaB=dim,
#                       dim_thetaC=0,initial_params=initial_params,
#                       name='RBF_network') 
    

    
#     identification_results = param_optim.ModelTraining(model,data,10,
#                              initial_params=initial_params,p_opts=None,
#                              s_opts=None)

#     pkl.dump(identification_results,open('Bioreactor_Lachhab_2states_theta'+str(dim)+'.pkl',
#                                          'wb'))
