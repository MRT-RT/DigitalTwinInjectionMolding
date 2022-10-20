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


path_dim = Path.cwd()
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous.PreProcessing import PIM_Data
from DIM.optim.param_optim import ModelTraining

class model_bank():
    def __init__(self,model_paths):
        self.model_paths = model_paths
        
        self.load_models()
        
        self.loss = [np.nan for m in self.models]
        self.pred = [None for m in self.models]
    
    def load_models(self):
        
        self.models = [pkl.load(open(path,'rb')) for path in self.model_paths]
        
        
def config_data_manager(source_hdf5,target_hdf5):
    # source_hdf5 = Path('/home/alexander/Downloads/Temperaturgangmessung-20221002.h5')
    
    
    # source_hdf5 = Path('C:/Users/LocalAdmin/Documents/DIM_Data/Messung 5.10/DIM_Temperaturgang_fixed.h5')
    # source_hdf5 = Path.cwd()/'live_data.h5'
    # source_hdf5 = Path('C:/Users/LocalAdmin/Documents/DIM_Data/Messung 6.10/Temperaturgangmessung-20221004.h5')
    # source_hdf5 = Path('C:/Users/LocalAdmin/Documents/DIM_Data/Messung 7.10/Temperaturgangmessung-20221005.h5')
    
    # target_hdf5 = Path.cwd()/'TGang_051022.h5'
    # target_hdf5 = Path.cwd()/'TGang_061022.h5'
    # target_hdf5 = Path.cwd()/'TGang_071022.h5'
    
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
    setpoints = ['T_zyl5_soll','v_inj_soll','V_um_soll','T_wkz_soll']
    
    # initialize data reader
    data_manager = PIM_Data(source_hdf5,target_hdf5,charts,scalar,scalar_dtype,
                           features,features_dtype,quals,quals_dtype,setpoints)
    
    return data_manager
    
def predict_quality(data_manager, model_bank):
    
    mod_data = pd.read_hdf(data_manager.target_hdf5, 'modelling_data')
    
    for m in range(len(model_bank.models)):
        
        model = model_bank.models[m]
        
        loss,pred = model.static_mode(mod_data)
        
        model_bank.loss[m] = loss
        model_bank.pred[m] = pred
        
    # Get "identification" data
    
    # Do a prediction for every model in the model_bank
    
    # Plot best prediction and info which model did it
    
    # sleep a while
    
    # plt.pause(0.0001)
    
    return None

def reestimate_models(ident_data, model,name):
        
    res = ModelTraining(model,ident_data,ident_data,initializations=1,
                        mode='static')
    
    model.Parameters = res.loc[0,'params_train']
    
    print('Finish '+str(m),flush=True)
    
    pkl.dump(model,open(name+'.mod'))
    # while loop infinite
    
    # Check if reader_status is True
    
    # read in model bank from a file
    
    # Do a prediction for every model in the model_bank
    
    # Reestimate models whose error is CRITERIUM
    
    # Save models to file, mark best model
    
    return None
 

    