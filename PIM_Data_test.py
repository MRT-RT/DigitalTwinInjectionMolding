#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:02:48 2022

@author: alexander
"""

from pathlib import Path
import sys
import h5py
import pandas as pd

path_dim = Path.cwd()
sys.path.insert(0, path_dim.as_posix())


from DIM.miscellaneous.PreProcessing import PIM_Data


# source_hdf5 = 'C:\Users\alexa\Downloads\data\Prozessgrößen_20211005.h5'

source_hdf5 = Path('/home/alexander/Downloads/Prozessgrößen_20211007.h5')

# source_hdf5 = Path('/home/alexander/Downloads/Prozessgrößen_20211006_1.h5')

target_hdf5 = 'test.h5'

charts = [{'keys':['f3103I_Value','f3203I_Value','f3303I_Value'],
           'values':['timestamp','Q_Vol_ist','None1']},
          
          {'keys':['f3113I_Value','f3213I_Value','f3313I_Value'],
           'values':['timestamp','p_wkz_ist', 'T_wkz_ist', 'p_inj_soll',
                     'p_inj_ist','None2']},
          
          {'keys':['f3403I_Value','f3503I_Value','f3603I_Value'],
           'values':['timestamp','V_Screw_ist','None3']}]

scalar = {'Q305_Value':'Einspritzgeschwindigkeit',
          'T801I_Value':'T_zyl1_ist',
          'T802I_Value':'T_zyl2_ist',
          'T803I_Value':'T_zyl3_ist',
          'T804I_Value':'T_zyl4_ist',
          'T805I_Value':'T_zyl5_ist',
          'V305_Value':'Umschaltpunkt',
          'V4065_Value':'V_um_ist',
          'p4072_Value':'p_um_ist',
          'p4055_Value':'p_inj_max_ist',
          'p312_Value':'Nachdruckhöhe',
          't4015_Value':'t_dos_ist',
          't4018_Value':'t_inj_ist',
          't312_Value':'Nachdruckzeit',
          't313_Value':'t_press2_soll',
          'f071_Value':'Zyklus',
          'p403_Value':'Staudruck'}

scalar_dtype = {'Einspritzgeschwindigkeit':'float16',
          'T_zyl1_ist':'float16',
          'T_zyl2_ist':'float16',
          'T_zyl3_ist':'float16',
          'T_zyl4_ist':'float16',
          'T_zyl5_ist':'float16',
          'Umschaltpunkt':'float16',
          'V_um_ist':'float16',
          'p_um_ist':'float16',
          'p_inj_max_ist':'float16',
          'Nachdruckhöhe':'float16',
          't_dos_ist':'float16',
          't_inj_ist':'float16',
          'Nachdruckzeit':'float16',
          't_press2_soll':'float16',
          'Zyklus':'int16',
          'Staudruck':'float16'}

features = ['T_wkz_0']
features_dtype = {'T_wkz_0':'float16'}


# initialize data reader
data_reader = PIM_Data(source_hdf5,target_hdf5,charts,scalar,scalar_dtype,
                       features,features_dtype)


# while True:
    
# Parse new data to target hdf5 if available
data_reader.get_cycle_data()

# Read data from target hdf5

df_overview = pd.read_hdf(target_hdf5,key='overview')
    
    
    
# open taget file
# target_file = h5py.File(target_hdf5,'r')

# read from target file


cycle_1756 = pd.read_hdf(target_hdf5,key='process_values/cycle_1756')
