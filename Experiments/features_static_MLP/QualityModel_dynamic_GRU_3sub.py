# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:44:37 2021

@author: alexa
"""

import pickle as pkl
import numpy as np

import multiprocessing

import sys
# sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/DigitalTwinInjectionMolding/')


from DIM.miscellaneous.PreProcessing import arrange_data_for_ident, eliminate_outliers
from DIM.models.model_structures import GRU
from DIM.models.injection_molding import QualityModel
from DIM.optim.param_optim import ModelTraining, parallel_mode
from DIM.miscellaneous.PreProcessing import LoadDynamicData
from DIM.optim.common import BestFitRate

# path = 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
# path = '/home/alexander/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/'
path = 'E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/normalized/'

charges = list(range(1,275))

dim_c = 9

split = 'all'
mode='quality'
    
   
u_inj= ['p_wkz_ist','T_wkz_ist']
u_press= ['p_wkz_ist','T_wkz_ist']
u_cool= ['p_wkz_ist','T_wkz_ist']

u_lab = [u_inj,u_press,u_cool]
y_lab = ['Durchmesser_innen']

# norm_cycle = pkl.load(open(path+'cycle1.pkl','rb'))

data_train,data_val = \
LoadDynamicData(path,charges,split,y_lab,u_lab,mode)

  
c0_train = [np.zeros((dim_c,1)) for i in range(0,len(data_train['data']))]
c0_val = [np.zeros((dim_c,1)) for i in range(0,len(data_val['data']))]    




inj_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,
                u_label=u_inj,y_label=y_lab,dim_out=1,name='inj')

press_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=1,
                  u_label=u_press,y_label=y_lab,dim_out=1,name='press')

cool_model = GRU(dim_u=2,dim_c=dim_c,dim_hidden=10,
                  u_label=u_cool,y_label=y_lab,dim_out=1,name='cool')

press_model.InitialParameters ={'b_z_press':np.random.uniform(-10,-4,(dim_c,1))}
cool_model.InitialParameters = {'b_z_cool':np.random.uniform(-10,-4,(dim_c,1))}

model_q = QualityModel(subsystems=[inj_model,press_model,cool_model],
                              name='q_model')

results_q = pkl.load(open('GRU_c9_3sub_all.pkl','rb'))

params = results_q.loc[10]['params_val']

model_q.SetParameters(params)

_,prediction_q = parallel_mode(model_q,data_val)


# s_opts = {"max_iter": 1000, 'hessian_approximation':'limited-memory'}


# results_GRU = ModelTraining(quality_model,data_train,data_val,initializations=20, BFR=False, 
#                   p_opts=None, s_opts=s_opts,mode='parallel')

# print(BestFitRate([c.loc[0][y_lab].values for c in data_val['data']],
#                   [c[y_lab].values for c in prediction_q]))

# fig, ax = plt.subplots(figsize=(20, 10))
# sns.stripplot(x=data_val.index,y=data_val['p_wkz_max'],color='grey',alpha=.8,
#               size=15,ax=ax)
# sns.stripplot(x=prediction_p.index,y=prediction_p['p_wkz_max'],size=15,ax=ax)
# ax.set_xlim([1,50]) 

