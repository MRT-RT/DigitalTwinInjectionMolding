#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 16:56:21 2022

@author: alexander
"""

from pathlib import Path
import sys
import h5py
import pickle as pkl
import time
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import seaborn as sns
from multiprocessing import Process, Pool
import copy


path_dim = Path.cwd()
sys.path.insert(0, path_dim.as_posix())

from DIM.arburg470.data_manager import Data_Manager
from DIM.optim.param_optim import ParamOptimizer

from DIM.optim.control_optim import StaticProcessOptimizer

class model_bank():
    def __init__(self,model_paths):
        self.model_paths = model_paths
        
        self.load_models()
        
        self.stp_loss = [np.nan for m in self.models]
        self.stp_pred = [None for m in self.models]
        
        self.rec_pred = [None for m in self.models]
        
    
    def load_models(self):
        
        self.models = [pkl.load(open(path,'rb')) for path in self.model_paths]
        
        
def config_data_manager(source_hdf5,target_hdf5,setpoints):

    
    charts = [{'keys':['f3113I_Value','f3213I_Value','f3313I_Value'],
               'values':['p_wkz_ist','p_hyd_ist','T_wkz_ist','p_hyd_soll',
                         'state1']},
              {'keys':['f3103I_Value','f3203I_Value','f3303I_Value'],
               'values':['V_screw_ist','state2']},
              {'keys':['f3403I_Value','f3503I_Value','f3603I_Value'],
               'values':['Q_inj_ist','state3']}
              ]
    
    
    scalar = {'T801I_Value':'T_zyl1_ist',
              'T802I_Value':'T_zyl2_ist',
              'T803I_Value':'T_zyl3_ist',
              'T804I_Value':'T_zyl4_ist',
              'T805I_Value':'T_zyl5_ist',
              'T801_Value':'T_zyl1_soll',
              'T802_Value':'T_zyl2_soll',
              'T803_Value':'T_zyl3_soll',
              'T804_Value':'T_zyl4_soll',
              'T805_Value':'T_zyl5_soll',
              'V305_Value':'V_um_soll',
              'V4065_Value':'V_um_ist',
              'V301I_Value':'V_dos_ist',
              'V403_Value':'V_dos_soll',
              'Q305_Value':'v_inj_soll',
              'p311_Value':'p_pack1_soll',
              'p312_Value':'p_pack2_soll',
              'p313_Value':'p_pack3_soll',
              't311_Value':'t_pack1_soll',
              't312_Value':'t_pack2_soll',
              't313_Value':'t_pack3_soll',
              'p403_Value':'p_stau_soll',
              'p4072_Value':'p_um_ist',
              'p4055_Value':'p_max_ist',
              't007_Value':'Uhrzeit',
              't4012_Value':'t_zyklus_ist',
              't4015_Value':'t_dos_ist',
              't4018_Value':'t_inj_ist',
              't400_Value':'t_cool_soll',
              'f071_Value': 'Zyklus',
               'T_wkz_soll': 'T_wkz_soll'}
    
    scalar_dtype = {'T_zyl1_ist':'float16',
                    'T_zyl2_ist':'float16',
                    'T_zyl3_ist':'float16',
                    'T_zyl4_ist':'float16',
                    'T_zyl5_ist':'float16',
                    'T_zyl1_soll':'float16',
                    'T_zyl2_soll':'float16',
                    'T_zyl3_soll':'float16',
                    'T_zyl4_soll':'float16',
                    'T_zyl5_soll':'float16',
                    'V_um_soll':'float16',
                    'V_um_ist':'float16',
                    'V_dos_ist':'float16',
                    'V_dos_soll':'float16',
                    'v_inj_soll':'float16',
                    'p_pack1_soll':'float16',
                    'p_pack2_soll':'float16',
                    'p_pack3_soll':'float16',
                    't_pack1_soll':'float16',
                    't_pack2_soll':'float16',
                    't_pack3_soll':'float16',
                    'p_stau_soll':'float16',
                    'p_um_ist':'float16',
                    'p_max_ist':'float16',
                    'Uhrzeit': 'datetime64[ns]',
                    't_zyklus_ist':'float16',
                    't_dos_ist':'float16',
                    't_inj_ist':'float16',
                    't_cool_soll':'float16',
                    'Zyklus':'int16',
                    'T_wkz_soll':'float16'}
    
    features = ['T_wkz_0']
    features_dtype = {'T_wkz_0':'float16'}
    
    quals = ['Messzeit', 'Losnummer', 'laufenden Zähler', 'OK/N.i.O.', 'Nummer',
           'Durchmesser_innen', 'Durchmesser_außen', 'Stegbreite_Gelenk',
           'Breite_Lasche', 'Rundheit_außen', 'Gewicht', 'ProjError']
    
    quals_dtype = {'Messzeit':'datetime64',
                   'Losnummer':'float16',
                   'laufenden Zähler':'int16',
                   'OK/N.i.O.':'bool',
                   'Nummer':'int16',
                   'Durchmesser_innen':'float16',
                   'Durchmesser_außen':'float16',
                   'Stegbreite_Gelenk':'float16',
                   'Breite_Lasche':'float16',
                   'Rundheit_außen':'float16',
                   'Gewicht':'float16',
                   'ProjError':'bool'}
    
    # Process/machine parameters that can be influenced by the operator
    setpoints = setpoints
    
    # initialize data reader
    dm = Data_Manager(source_hdf5,target_hdf5,charts,scalar,scalar_dtype,
                           features,features_dtype,quals,quals_dtype,setpoints)
    
    return dm
    
def predict_quality(data_manager, model_bank):
    
    dm = data_manager
    mb = model_bank
    
    # Load managed setpoint data    
    mod_data = pd.read_hdf(dm.target_hdf5, 'modelling_data')
    
    # load most recent data, solely for plotting purposes
    df_scalar = pd.read_hdf(dm.target_hdf5,'overview')
    df_feat = pd.read_hdf(dm.target_hdf5,'features')
    df_qual = pd.read_hdf(dm.target_hdf5,'quality_meas')
    
    # Sort dataframe
    df_scalar = df_scalar.sort_index()
    
    # find most recent observations
    idx_rec = df_scalar.index[-20::]
    
    rec_data = pd.concat([df_scalar.loc[idx_rec],
                          df_feat.loc[idx_rec],
                          df_qual.loc[idx_rec]],axis=1)
        
    for m in range(len(mb.models)):
        
        
        # 1: Prediction of managed setpoint data. 
        # predict data for managed observations over all setpoints
        model = mb.models[m]
        
        # Normalize data
        norm_data = model.MinMaxScale(mod_data)
        
        # Predict quality
        loss,pred = model.static_mode(norm_data)
        
        # Reverse normalization
        pred_un = model.MinMaxScale(pred,reverse=True)        
        
        # Rename columns
        pred_cols = [col+'_pred' for col in pred_un.columns]
        pred_un.columns = pred_cols
        
        # Concatenate measured data and prediction
        stp_pred = pd.concat([mod_data,pred_un],axis=1)
        
        mb.stp_loss[m] = loss
        mb.stp_pred[m] = stp_pred
        
        
        # 2: Predcition over most recent 20 observations, solely for plotting
        # purposes
        norm_data = model.MinMaxScale(rec_data)

        # Predict quality
        _,pred = model.static_mode(norm_data)
        
        # Reverse normalization
        pred_un = model.MinMaxScale(pred,reverse=True)        
        
        # Rename columns
        pred_cols = [col+'_pred' for col in pred_un.columns]
        pred_un.columns = pred_cols
        
        # Concatenate measured data and prediction
        rec_pred = pd.concat([rec_data,pred_un],axis=1)
        
        mb.rec_pred[m] = rec_pred
        
    
    return None

def estimate_parallel(model,data,opts,path):
    """
    Helper function for reestimate_models() to realize parallelization

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    opts : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    data_norm = model.MinMaxScale(data)
    param_optimizer = ParamOptimizer(model,data_norm,data_norm,**opts)
    res = param_optimizer.optimize()
    model.Parameters = res.loc[0,'params_train']
    pkl.dump(model,open(path,'wb'))        
    print('Finish '+str(model.name),flush=True)

def reestimate_models(data_manager, model_bank):
    
    dm = data_manager
    mb = model_bank
    
    ident_data = pd.read_hdf(dm.target_hdf5,key='modelling_data')
   
    opts = {'initializations':1,
            's_opts':{"max_iter": 100, "print_level":1, 
                     "hessian_approximation":'limited-memory'},
            'mode' : 'static'}
    
    # Start a process for reestimation of each model 
    for m in range(len(mb.models)):
        p = Process(target=estimate_parallel,
                    args=(mb.models[m],ident_data,opts,mb.model_paths[m]))
        
        p.start()
        p.join()
    
    return None

def optimize_parallel(model,Q_target,fix_inputs,init_values,constraints):
    
    setpoint_Optimizer = StaticProcessOptimizer(model=model)
    U_sol = setpoint_Optimizer.optimize(Q_target,fix_inputs,
                                input_init=init_values,
                                constraints=constraints)
    U_sol_norm = model.MinMaxScale(U_sol,reverse=True)
    
    return U_sol_norm
    

def optimize_setpoints(data_manager,model_bank,Q_target):

    dm = data_manager
    mb = model_bank
    
    # find best overall model
    idx_mod = np.argmin(mb.stp_loss)
    model = mb.models[idx_mod]

    # load data used for modelling 
    data = pd.read_hdf(dm.target_hdf5,key='modelling_data')
    
    # normalize data
    data = model.MinMaxScale(data)
    
    # Get T_wkz_0 from most recent measurement
    fix_inputs = data.loc[[data.index[-1]],['T_wkz_0']]
        
    # find unique setpoints and select only model inputs
    stp_data = data.drop_duplicates(subset='Setpoint')
    inputs = stp_data[model.u_label]
    
    # remove model inputs that can't be influenced
    man_inputs = inputs.drop(columns=['T_wkz_0'])
    
    # derive constraints and initial values from data used for modelling 
    init_values = [man_inputs.loc[[k]] for k in man_inputs.index]
        
    # get constraints from data
    constraints = []
    for u in man_inputs.columns:
        
        col_min = man_inputs[u].min()
        col_max = man_inputs[u].max()
        
        if col_min == col_max:
            col_min = col_min*0.9
            col_max = col_max*1.1

        constraints.append((u,'>'+str(col_min)))
        constraints.append((u,'<'+str(col_max)))             
    
    
    # Copy everything for parallel pool
    n_init = len(init_values)
    
    model = [copy.deepcopy(model) for i in range(n_init)]
    Q_target = [copy.deepcopy(Q_target) for i in range(n_init)]
    fix_inputs = [copy.deepcopy(fix_inputs) for i in range(n_init)]
    constraints = [copy.deepcopy(constraints) for i in range(n_init)]
   
    pool = Pool(n_init)
    
    results = pool.starmap(optimize_parallel, zip(model, Q_target, fix_inputs,
                                                  init_values,constraints))       
    # close pool
    pool.close() 
    pool.join()  
    
    # cast results to pandas dataframe
    results = pd.concat(results,axis=1)
    
    # Enumerate solutions for plotting purposes
    results['Sol_Num'] = range(len(results))
    return results   

def plot_meas_pred(fig,ax,data_manager,model_bank):
    
    dm = data_manager
    mb = model_bank
      
    # Plot 1: Quality over Temperature in current setpoint
    # find best model and load prediction
    mod_idx = np.argmin(mb.stp_loss)            
    pred_spt = mb.stp_pred[mod_idx]
    
    # find current setpoint (look for last measurement added)
    spt = pred_spt.loc[max(pred_spt.index),'Setpoint']
    
    # find all cycles of that setpoint
    cyc_idx = (pred_spt['Setpoint']==spt).index    

    #plot setpoint data and prediction
    ax[0].cla()     # Clear axis
    opts = {'marker':'d','markersize':20}
    
    sns.lineplot(data=pred_spt.loc[cyc_idx],x = 'T_wkz_0',
                 y = 'Durchmesser_innen',ax=ax[0], color='k',**opts) 
    sns.lineplot(data=pred_spt.loc[cyc_idx],x = 'T_wkz_0',
                 y = 'Durchmesser_innen_pred',ax=ax[0], color='b',**opts)             
    
    
    # Plot 2: Quality over cycle number (last 20)
    pred_rec = mb.rec_pred[mod_idx]

    ax[1].cla()     # Clear axis
    sns.lineplot(data=pred_rec,x = pred_rec.index,
                 y = 'Durchmesser_innen',ax=ax[1],color='k',**opts) 
    sns.lineplot(data=pred_rec,x = pred_rec.index,
                 y = 'Durchmesser_innen_pred',ax=ax[1],color='b',**opts)
    ax[1].set_xticks(pred_rec.index)

    
    ax[0].set_xlabel('T_wkz_0',fontsize = 22)
    ax[1].set_xlabel('Zyklus',fontsize = 22)
    
    ax[0].set_ylabel('Durchmesser_innen',fontsize = 22)
    ax[1].set_ylabel('Durchmesser_innen',fontsize = 22)
    
    
    
    [a.set_ylim([27,28]) for a in ax]
    plt.pause(0.01)
    # fig.tight_layout()
    # fig.canvas.flush_events() 
    # plt.pause(0.01)
    
def parallel_plot_setpoints(fig,ax,opti_setpoints):
    
    [a.cla() for a in ax]     # Clear axis
    
    # Calculate number of different solutions
    n_sol = len(opti_setpoints)
    
    # get as many different colors
    col_pal = sns.color_palette(n_colors = n_sol)
    
    # 
    cols = opti_setpoints.columns
    
    order = [n for n in opti_setpoints['Sol_Num']]
    
    for c in range(len(cols)-1):
        
        col = cols[c]
        
        p = sns.stripplot(data=opti_setpoints,
                     hue='Sol_Num',
                     x=np.arange(0,1,1/n_sol),
                     y=col,
                     ax=ax[c],
                     size=20,
                     palette = col_pal,
                     dodge=False)
                     # order=order)
                
        ax[c].set_ylim([opti_setpoints[col].min()*0.95,
                        opti_setpoints[col].max()*1.05])
        
        ax[c].set_yticks(opti_setpoints[col])
        
        ax[c].set_xticklabels([])
        ax[c].set_xticks([])
        
        ax[c].set_title(col)
        ax[c].set_ylabel(None)
        ax[c].grid(axis='y')
        
        legend = ax[c].legend()
        legend.remove()        
        

    plt.pause(0.01)
    # fig.tight_layout()
    # fig.canvas.flush_events() 
    # plt.pause(0.01)
    # plt.show(block=False)
    

    
    