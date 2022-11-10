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
from DIM.optim.common import BestFitRate

class model_bank():
    def __init__(self,model_path):
        self.model_path = model_path
        
        self.load_models()
        
        self.stp_loss = [np.nan for m in self.models]
        self.stp_pred = [None for m in self.models]
        
        self.rec_pred = [None for m in self.models]
        self.stp_bfr = [None for m in self.models]
        
    
    def load_models(self):
        
        model_dict = pkl.load(open(self.model_path,'rb'))
        
        self.models = [model_dict[i]['val'] for i in model_dict.keys()]

class ModelQualityPlot():
    def __init__(self): 
        self.fig,self.ax = plt.subplots(1,1)
        self.mngr = plt.get_current_fig_manager()
        self.mngr.window.setGeometry(1920//4*2, 530, 1920//4 , 500) 

        self.memory = 10


        self.fig.suptitle('Modellgüte')
        init_data = np.zeros((self.memory,1))
        self.plot_data = plt.plot(np.arange(-self.memory,0,1),
                                  init_data,marker='o')
        
        
        self.ax.set_ylabel('BFR in %')
        
        
        
        self.ax.set_ylim([0,100])       
        self.fig.tight_layout()
        
    def update(self,bfr):
        
        pd = self.plot_data[0]
        dx,dy = pd.get_data()
        
        dy = np.hstack((dy,np.array([bfr])))
        dy = dy[1::]
        
        pd.set_data((dx,dy))
        
        self.fig.tight_layout()
        self.fig.canvas.draw()


class SolutionQualityPlot():
    def __init__(self): 
        self.fig,self.ax = plt.subplots(1,1)
        self.mngr = plt.get_current_fig_manager()
        self.mngr.window.setGeometry(1920//4*3, 530, 1920//4 , 500) 

        self.memory = 10

        self.fig.suptitle('Lösungsgüte')
        init_data = np.zeros((self.memory,1))
        self.plot_data = plt.plot(np.arange(-self.memory,0,1),
                                  init_data,marker='o')
        
        self.fig.canvas.draw()
        
    def update(self,e):
        
        pd = self.plot_data[0]
        dx,dy = pd.get_data()
        
        dy = np.hstack((dy,np.array([e])))
        dy = dy[1::]
        
        pd.set_data((dx,dy))
        
        self.ax.set_ylim([min(dy)*0.99,max(dy)*1.09])
        
        self.fig.tight_layout()
        self.fig.canvas.draw()
        
class PredictionPlot():
    def __init__(self): 
        
        self.fig,self.ax = plt.subplots(1,2)
        
        self.mngr = plt.get_current_fig_manager()
        
        self.mngr.window.setGeometry(0, 30, 1920 , 500)

        self.fig.suptitle('Qualitätsmessung und -prädiktion')

        init_data = np.zeros((20,1))
        
        opt_meas = {'marker':'d','markersize':20,'linestyle':'none'}
        opt_pred = {'marker':'d','markersize':15,'linestyle':'none'}
        
        self.meas_data_1 = self.ax[0].plot(range(0,20),init_data,
                                           **opt_meas)
        
        self.pred_data_1 = self.ax[0].plot(range(0,20),init_data,
                                           **opt_pred)
        
        self.ax[0].set_xlabel('T in °C')
        self.ax[0].set_ylabel('m in g')
        
        self.meas_data_2 = self.ax[1].plot(range(0,20),init_data,
                                           **opt_meas)
        
        self.pred_data_2 = self.ax[1].plot(range(0,20),init_data,
                                           **opt_pred)
        
        self.ax[1].set_xlabel('Zyklus')
        self.ax[1].legend(['Messung','Prädiktion'])
        
        self.fig.tight_layout()
            
    def update(self,dm,mb):
        
        y_label = mb.models[0].y_label[0]
        
        # Update measurement data in setpoint plot
        d = self.meas_data_1[0]
        dx,dy = d.get_data()
        
        # load setpoint prediction for best model
        pred_spt = mb.stp_pred[mb.idx_best]

        # find current setpoint (look for last measurement added)
        spt = pred_spt.loc[max(pred_spt.index),'Setpoint']
        
        # find all cycles of that setpoint
        cyc_idx = (pred_spt['Setpoint']==spt)  
        
        dx = pred_spt.loc[cyc_idx]['T_wkz_0'].values
        dy = pred_spt.loc[cyc_idx][y_label].values       
        d.set_data((dx,dy))
        
        # Update pred data in setpoint plot
        d = self.pred_data_1[0]
        dx,dy = d.get_data()
        
        
        dx = pred_spt.loc[cyc_idx]['T_wkz_0'].values
        dy = pred_spt.loc[cyc_idx][y_label+'_pred'].values       
        d.set_data((dx,dy))

        self.ax[0].set_xlim([min(dx)*0.99,max(dx)*1.01])
        self.ax[0].set_ylim([min(dy)*0.99,max(dy)*1.01])

        # Plot 2: Quality over cycle number (last 20)
        pred_rec = mb.rec_pred[mb.idx_best]
        
        d = self.meas_data_2[0]
        dx,dy = d.get_data()
        
        dx = pred_rec.index.values
        dy = pred_rec[y_label].values     
        d.set_data((dx,dy))
        
        d = self.pred_data_2[0]
        dx,dy = d.get_data()
        
        dx = pred_rec.index.values
        dy = pred_rec[y_label+'_pred'].values     
        d.set_data((dx,dy))      
    
        self.ax[1].set_xlim([min(dx)-1,max(dx)+1])
        self.ax[1].set_ylim([min(dy)*0.99,max(dy)*1.01])
        
        self.fig.tight_layout()
        self.fig.canvas.draw()

        
class OptimSetpointsPlot():
    
    def __init__(self,num_sol): 
        
        self.fig,self.ax = plt.subplots(1,3)
        self.mngr = plt.get_current_fig_manager()
        self.mngr.window.setGeometry(0, 530, 1920//2 , 500) 
        
        self.num_sol = num_sol
        self.base_marker_size = 30
        
        self.fig.suptitle('Optimale Maschinenparameter')
        
        # self.plot_data_1 = [self.ax[0].scatter([0],[0],marker='o') for i in range(num_sol)]
        # self.plot_data_2 = self.ax[1].scatter([0]*num_sol,[0]*num_sol,marker='o')
        # self.plot_data_3 = self.ax[2].scatter([0]*num_sol,[0]*num_sol,marker='o')
        
        opts = {'marker':'o','alpha':0.9}
        
        self.plot_data_1 = [self.ax[0].scatter([0],[0],**opts) for i in range(num_sol)]
        self.plot_data_2 = [self.ax[1].scatter([0],[0],**opts) for i in range(num_sol)]
        self.plot_data_3 = [self.ax[2].scatter([0],[0],**opts) for i in range(num_sol)]
        
        self.hline_1 = self.ax[0].axhline(y = 0, color = 'r', linestyle = '-')
        self.hline_2 = self.ax[1].axhline(y = 0, color = 'r', linestyle = '-')
        self.hline_3 = self.ax[2].axhline(y = 0, color = 'r', linestyle = '-')
        
        # get_offsets() returns masked array the first time and an array after
        # that. So get and set here one time.
        for plot_data in [self.plot_data_1,self.plot_data_2,self.plot_data_3]:
            for sol in range(num_sol):
                d = plot_data[sol]
                ma = d.get_offsets()
                d.set_offsets(ma)
        
        self.fig.tight_layout()
        self.fig.canvas.draw()
            
    def update(self,opti_setpoints,stp):
        
        # loss limit 
        lim = 0.001
        
        # find bad solutions
        n_bad = sum(opti_setpoints['loss']>lim)
        if n_bad:
            print('Ignored ' + str(n_bad) + ' solutions that exceeded loss of 0.01.')
            
        # print('Lösungsgüte: ' + str(opti_setpoints['loss'].min()))    
        
        # # Keep only good solutions
        opti_setpoints = opti_setpoints.loc[opti_setpoints['loss']<=lim]
        opti_setpoints = opti_setpoints.drop(columns='loss')
        
        if opti_setpoints.empty:
            return None
        
        
        opti_norm = (opti_setpoints-opti_setpoints.mean())/opti_setpoints.std()
        
        # Aggregate similar solutions
        
        solutions = pd.DataFrame(columns=list(opti_setpoints.columns) + ['weight'],
                                 index = range(self.num_sol))
        
        for i in range(0,self.num_sol):
            j = opti_norm.index[0]

            diff = opti_norm.loc[j::]-opti_norm.loc[j]
            diff = diff.apply(np.linalg.norm,axis=1)
            
            idx = diff[diff<0.25].index
            
            sol = opti_setpoints.loc[idx].mean()
            
            solutions.loc[i] = sol
            solutions.loc[i]['weight'] = float(len(idx))
            
            opti_norm = opti_norm.drop(index=idx)
            
            if opti_norm.empty:
                break
        
        # Drop empty rows
        idx_nan = solutions.isna().any(axis=1)
        solutions.loc[idx_nan]=0.0
        
        # Sort solutions by proximity to current setpoint
        stp = stp.drop(columns='Gewicht')                                       # Replace Gewicht by y_label
        d1 =  solutions.drop(columns='weight').reset_index(drop=True)  
        d2 = stp.reset_index(drop=True)
        diff = d1-d2.values
        diff = diff.apply(np.linalg.norm,axis=1)
        sort_idx = diff.sort_values().index
        
        # Assign new data to plot
        plot_data = [self.plot_data_1,self.plot_data_2,self.plot_data_3]
        line_data = [self.hline_1,self.hline_2,self.hline_3]
        
        for p in range(3):
            col = solutions.columns[p]
            
            l = line_data[p]
            ld = ([0,1],[float(stp[col].values),float(stp[col].values)])
            l.set_data(ld)
            
            for sol in sort_idx:
                d = plot_data[p][sol]
                ma = d.get_offsets()
                ma[:,1] = solutions.loc[sol,col]
                d.set_offsets(ma)
                d.set_sizes([solutions.loc[sol,'weight']*self.base_marker_size])
            
            

            
            self.ax[p].set_ylim([solutions.loc[0:i,col].min()*0.98,
                                 solutions.loc[0:i,col].max()*1.02])
            
            self.ax[p].set_title(col)
        
            
        
        self.fig.tight_layout()
        self.fig.canvas.draw()  
        
        
        
def config_data_manager(source_hdf5,target_hdf5,setpoints):

    
    charts = [{'keys':['f3113I_Value','f3213I_Value','f3313I_Value'],
               'values':['p_wkz_ist','T_wkz_ist','p_hyd_soll','p_hyd_ist',
                         'state1']},
              {'keys':['f3103I_Value','f3203I_Value','f3303I_Value'],
               'values':['Q_inj_ist','state2']},
              {'keys':['f3403I_Value','f3503I_Value','f3603I_Value'],
               'values':['V_screw_ist','state3']}
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
               # 'f071_Value': 'Zyklus',
               'f9002_Value': 'Zyklus',
               'T822_Value': 'T_wkz_soll'}
    
    scalar_dtype = {'T_zyl1_ist':'float32',
                    'T_zyl2_ist':'float32',
                    'T_zyl3_ist':'float32',
                    'T_zyl4_ist':'float32',
                    'T_zyl5_ist':'float32',
                    'T_zyl1_soll':'float32',
                    'T_zyl2_soll':'float32',
                    'T_zyl3_soll':'float32',
                    'T_zyl4_soll':'float32',
                    'T_zyl5_soll':'float32',
                    'V_um_soll':'float32',
                    'V_um_ist':'float32',
                    'V_dos_ist':'float32',
                    'V_dos_soll':'float32',
                    'v_inj_soll':'float32',
                    'p_pack1_soll':'float32',
                    'p_pack2_soll':'float32',
                    'p_pack3_soll':'float32',
                    't_pack1_soll':'float32',
                    't_pack2_soll':'float32',
                    't_pack3_soll':'float32',
                    'p_stau_soll':'float32',
                    'p_um_ist':'float32',
                    'p_max_ist':'float32',
                    'Uhrzeit': 'datetime64[ns]',
                    't_zyklus_ist':'float32',
                    't_dos_ist':'float32',
                    't_inj_ist':'float32',
                    't_cool_soll':'float32',
                    'Zyklus':'int16',
                    'T_wkz_soll':'float32'}
    
    features = ['T_wkz_0']
    features_dtype = {'T_wkz_0':'float32'}
    
    quals = ['Messzeit', 'Losnummer', 'laufenden Zähler', 'OK/N.i.O.', 'Nummer',
           'Durchmesser_innen', 'Durchmesser_außen', 'Stegbreite_Gelenk',
           'Breite_Lasche', 'Rundheit_außen', 'Gewicht', 'ProjError']
    
    quals_dtype = {'Messzeit':'datetime64',
                   'Losnummer':'float32',
                   'laufenden Zähler':'int16',
                   'OK/N.i.O.':'bool',
                   'Nummer':'int16',
                   'Durchmesser_innen':'float32',
                   'Durchmesser_außen':'float32',
                   'Stegbreite_Gelenk':'float32',
                   'Breite_Lasche':'float32',
                   'Rundheit_außen':'float32',
                   'Gewicht':'float32',
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
        orig_cols = [col for col in pred_un.columns]
        pred_cols = [col+'_pred' for col in pred_un.columns]
        pred_un.columns = pred_cols       
        
        # Concatenate measured data and prediction
        stp_pred = pd.concat([mod_data,pred_un],axis=1)
        
        # Calculate Best Fit Rate
        bfr = BestFitRate(y_est=stp_pred[pred_cols],
                          y_target=stp_pred[orig_cols])
        
        mb.stp_loss[m] = loss
        mb.stp_pred[m] = stp_pred
        mb.stp_bfr[m] = bfr
        
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
        
    mb.idx_best = np.argmin(mb.stp_loss)
        
    
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
            's_opts':{"max_iter": 500, "print_level":1, 
                     "hessian_approximation":'limited-memory'},
            'mode' : 'static'}
    
    # Start a process for reestimation of each model 
    print('Models are reestimated...')
    for m in range(len(mb.models)):
        p = Process(target=estimate_parallel,
                    args=(mb.models[m],ident_data,opts,mb.model_paths[m]))
        
        p.start()
        p.join()
    print('Estimation complete.')
    return None

def optimize_parallel(model,Q_target,fix_inputs,init_values,constraints):
    
    setpoint_Optimizer = StaticProcessOptimizer(model=model)
    U_sol = setpoint_Optimizer.optimize(Q_target,
                                        fix_inputs=fix_inputs,
                                        input_init=init_values,
                                        constraints=constraints)
    
    U_sol_norm = model.MinMaxScale(U_sol,reverse=True)
    
    return U_sol_norm
    

def optimize_setpoints(data_manager,model_bank,Q_target,fix_labels):

    dm = data_manager
    mb = model_bank
    
    # find best overall model
    idx_mod = np.argmin(mb.stp_loss)
    model = mb.models[idx_mod]

    # load data used for modelling 
    data = pd.read_hdf(dm.target_hdf5,key='modelling_data')
    
    # normalize data
    data = model.MinMaxScale(data)
    Q_target = model.MinMaxScale(Q_target)
    
    # Get fixed input values from most recent measurement
    fix_inputs = data.loc[[data.index[-1]],fix_labels]
        
    # find unique setpoints and select only model inputs
    stp_data = data.drop_duplicates(subset='Setpoint')
    inputs = stp_data[model.u_label]
    
    # remove model inputs that can't be influenced
    man_inputs = inputs.drop(columns=fix_labels)
    
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
    
    
    print('Calculate optimal setpoints...')
    results = pool.starmap(optimize_parallel, zip(model, Q_target, fix_inputs,
                                                  init_values,constraints))       
    # close pool
    pool.close() 
    pool.join()  
    print('Optimal setpoints calculated')
    
    
    # cast results to pandas dataframe
    results = pd.concat(results,axis=0)
    
    # Enumerate solutions for plotting purposes
    results = results.sort_values(ascending=True,by='loss')
    results.index = range(len(results))
    
    return results   

def plot_meas_pred(figax1,figax2,data_manager,model_bank):
    
    fig1 = figax1[0]
    ax1 = figax1[1]
    
    fig2 = figax2[0]
    ax2 = figax2[1]
    
    dm = data_manager
    mb = model_bank
    
    y_label = mb.models[0].y_label[0]
      
    # Plot 1: Quality over Temperature in current setpoint
    # find best model and load prediction
    mod_idx = np.argmin(mb.stp_loss)            
    pred_spt = mb.stp_pred[mod_idx]
    
    # find current setpoint (look for last measurement added)
    spt = pred_spt.loc[max(pred_spt.index),'Setpoint']
    
    # find all cycles of that setpoint
    cyc_idx = (pred_spt['Setpoint']==spt)   

    #plot setpoint data and prediction
    ax1[0].cla()     # Clear axis
    opts1 = {'marker':'d','markersize':20}
    opts2 = {'marker':'d','markersize':15}
    
    sns.lineplot(data=pred_spt.loc[cyc_idx],x = 'T_wkz_0',
                 y = y_label,ax=ax1[0], color='k',**opts1) 
    sns.lineplot(data=pred_spt.loc[cyc_idx],x = 'T_wkz_0',
                 y = y_label+'_pred',ax=ax1[0], color='b',**opts2)             
    
    
    # Plot 2: Quality over cycle number (last 20)
    pred_rec = mb.rec_pred[mod_idx]

    ax1[1].cla()     # Clear axis
    sns.lineplot(data=pred_rec,x = pred_rec.index,
                 y = y_label,ax=ax1[1],color='k',**opts1) 
    sns.lineplot(data=pred_rec,x = pred_rec.index,
                 y = y_label+'_pred',ax=ax1[1],color='b',**opts2)
    ax1[1].set_xticks(pred_rec.index[0::2])

    
    ax1[0].set_xlabel('T_wkz_0',fontsize = 12)
    ax1[1].set_xlabel('Zyklus',fontsize = 12)
    
    ax1[0].set_ylabel(y_label,fontsize = 12)
    ax1[1].set_ylabel(y_label,fontsize = 12)
    
    
    
    # [a.set_ylim([27,28]) for a in ax]
    
    y_min0 = pred_spt[y_label].min()*0.99
    y_max0 = pred_spt[y_label].max()*1.01
    
    y_min0 = pred_rec[y_label].min()*0.99
    y_max0 = pred_rec[y_label].max()*1.01
    
    ax1[0].set_ylim([y_min0,y_max0])
    ax1[1].set_ylim([y_min0,y_max0])
    fig1.tight_layout()
    plt.pause(0.01)
    fig1.canvas.draw()
    
def parallel_plot_setpoints(fig,ax,opti_setpoints):
    
    [a.cla() for a in ax]     # Clear axis
    
    # Calculate number of different solutions
    n_sol = len(opti_setpoints)
    
    # get as many different colors
    col_pal = sns.color_palette(n_colors = n_sol)
    
    # 
    cols = opti_setpoints.columns
    
    opti_setpoints['Sol_Num'] = range(n_sol)
    
    order = [n for n in opti_setpoints['Sol_Num']]
    
    for c in range(len(cols)):
        
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
                
        ax[c].set_ylim([opti_setpoints[col].min()*0.99,
                        opti_setpoints[col].max()*1.01])
        
        ax[c].set_yticks(opti_setpoints[col])
        
        ax[c].set_xticklabels([])
        ax[c].set_xticks([])
        
        ax[c].set_title(col)
        ax[c].set_ylabel(None)
        ax[c].grid(axis='y')
        
        legend = ax[c].legend()
        legend.remove()        
        

    fig.tight_layout()
    plt.pause(0.01)
    fig.canvas.draw()
    

    
    