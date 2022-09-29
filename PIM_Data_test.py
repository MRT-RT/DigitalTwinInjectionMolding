#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:02:48 2022

@author: alexander
"""

from pathlib import Path
import sys

path_dim = Path.cwd()
sys.path.insert(0, path_dim.as_posix())


from DIM.miscellaneous.PreProcessing import PIM_Data


# source_hdf5 = 'C:\Users\alexa\Downloads\data\Prozessgrößen_20211005.h5'

source_hdf5 = Path('C:/Users/alexa/Downloads/data/Prozessgrößen_20211007.h5')

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


data_reader = PIM_Data(source_hdf5,target_hdf5,charts,scalar)

charts,scalars = data_reader.get_cycle_data()