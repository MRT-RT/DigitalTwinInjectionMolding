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

path_dim = Path.cwd()
sys.path.insert(0, path_dim.as_posix())


from DIM.miscellaneous.PreProcessing import PIM_Data


# source_hdf5 = Path('/home/alexander/Downloads/Temperaturgangmessung-20221002.h5')

source_hdf5 = Path('C:/Users/LocalAdmin/Downloads/Temperaturgangmessung-20221002.h5')

# source_hdf5 = Path('Y:/Klute/DIM_Temperaturgang/OPC_UA_Arburg/Temperaturgangmessung-20221003.h5')

target_hdf5 = 'test.h5'

charts = [{'keys':['f3113I_Value','f3213I_Value','f3313I_Value'],
           'values':['timestamp','p_wkz_ist','p_hyd_ist','T_wkz_ist','p_hyd_soll',
                     'state1']},
          {'keys':['f3103I_Value','f3203I_Value','f3303I_Value'],
           'values':['timestamp','V_screw_ist','state2']},
          {'keys':['f3403I_Value','f3503I_Value','f3603I_Value'],
           'values':['timestamp','Q_inj_ist','state3']}
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
          'f071_Value': 'Zyklus'}

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
                'Zyklus':'int16'}

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



# initialize data reader
data_reader = PIM_Data(source_hdf5,target_hdf5,charts,scalar,scalar_dtype,
                       features,features_dtype,quals,quals_dtype)

pkl.dump(data_reader,open('Arburg_data_reader.pkl','wb'))