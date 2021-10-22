# -*- coding: utf-8 -*-

import h5py    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/home/alexander/Downloads/Versuchsplan/'

filename = 'Prozessgrößen_20211005.h5'


file = h5py.File(path+filename,'r+') 


def hdf5_to_pd_dataframe(file):
    '''
    

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    # print(file)
    for cycle in file.keys():
        

        # monitoring charts 1-3
        f3103I = file[cycle]['f3103I_Value']['block0_values'][:,[0,1]]
        f3203I = file[cycle]['f3203I_Value']['block0_values'][:,[0,1]]
        f3303I = file[cycle]['f3303I_Value']['block0_values'][:,[0,1]]    
        timestamp1 = np.vstack([f3103I[:,[0]],f3203I[:,[0]],f3303I[:,[0]]])
           
        # monitoring charts 4-6
        f3403I = file[cycle]['f3403I_Value']['block0_values'][:,[0,1]]
        f3503I = file[cycle]['f3503I_Value']['block0_values'][:,[0,1]]
        f3603I = file[cycle]['f3603I_Value']['block0_values'][:,[0,1]]  
        timestamp2 = np.vstack([f3403I[:,[0]],f3503I[:,[0]],f3603I[:,[0]]])
        
        # measuring chart
        f3113I = file[cycle]['f3113I_Value']['block0_values'][:,0:5]
        f3213I = file[cycle]['f3213I_Value']['block0_values'][:,0:5]
        f3313I = file[cycle]['f3313I_Value']['block0_values'][:,0:5]
        timestamp3 = np.vstack([f3113I[:,[0]],f3213I[:,[0]],f3313I[:,[0]]])       
        
        MonChart1_3 = np.vstack((f3103I,f3203I,f3303I)) 
        MonChart4_6 = np.vstack((f3403I,f3503I,f3603I)) 
        MeasChart = np.vstack((f3113I,f3213I,f3313I)) 
                     
        df1 = pd.DataFrame(data=MonChart1_3[:,[1]],
        index = timestamp1[:,0], columns = ['Q_Vol_ist'])
        
        df2 = pd.DataFrame(data=MonChart4_6[:,[1]],
        index = timestamp2[:,0], columns = ['V_Screw_soll'])
        
        cols = ['p_wkz_i', 'T_wkz_i', 'p_inj_soll','p_inj_ist']

        df3 = pd.DataFrame(data=MeasChart[:,1:5],
        index = timestamp3[:,0], columns = cols)
        
        df = pd.concat([df1,df2,df3],axis=1)
        
        # now add scalar values
        
        Q_inj_soll = file[cycle]['Q305_Value']['block0_values'][:]
        Q_inj_soll = pd.Series(np.repeat(Q_inj_soll,len(df)))
        df=df.assign(Q_inj_soll = Q_inj_soll.values)
        
        T_zyl_1 = file[cycle]['T801I_Value']['block0_values'][:]
        T_zyl_1 = pd.Series(np.repeat(T_zyl_1,len(df)))
        df=df.assign(T_zyl_1 = T_zyl_1.values)     
        
        T_zyl_2 = file[cycle]['T802I_Value']['block0_values'][:]
        T_zyl_2 = pd.Series(np.repeat(T_zyl_2,len(df)))
        df=df.assign(T_zyl_2 = T_zyl_2.values)   
        
        T_zyl_3 = file[cycle]['T803I_Value']['block0_values'][:]
        T_zyl_3 = pd.Series(np.repeat(T_zyl_3,len(df)))
        df=df.assign(T_zyl_3 = T_zyl_3.values)   
        
        T_zyl_4 = file[cycle]['T804I_Value']['block0_values'][:]
        T_zyl_4 = pd.Series(np.repeat(T_zyl_4,len(df)))
        df=df.assign(T_zyl_4 = T_zyl_4.values)   
        
        T_zyl_5 = file[cycle]['T805I_Value']['block0_values'][:]
        T_zyl_5 = pd.Series(np.repeat(T_zyl_5,len(df)))
        df=df.assign(T_zyl_5 = T_zyl_5.values)           
        
        
        
        
        
        cycle_num = file[cycle]['f071_Value']['block0_values'][0,0]
        
        # print(df)
        break
    
    return df
        
        
        
        
        
df = hdf5_to_pd_dataframe(file)        


