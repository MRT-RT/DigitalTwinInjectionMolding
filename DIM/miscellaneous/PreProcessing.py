# -*- coding: utf-8 -*-

  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl


def hdf5_to_pd_dataframe(file,save_path=None):
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
        
        try:
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
            index = timestamp2[:,0], columns = ['V_Screw_ist'])
            
            cols = ['p_wkz_ist', 'T_wkz_ist', 'p_inj_soll','p_inj_ist']
    
            df3 = pd.DataFrame(data=MeasChart[:,1:5],
            index = timestamp3[:,0], columns = cols)
            
            df = pd.concat([df1,df2,df3],axis=1)
            
            # now add scalar values
            
            Q_inj_soll = file[cycle]['Q305_Value']['block0_values'][:]
            Q_inj_soll = pd.Series(np.repeat(Q_inj_soll,len(df)))
            df=df.assign(Q_inj_soll = Q_inj_soll.values)
            
            T_zyl1_ist = file[cycle]['T801I_Value']['block0_values'][:]
            T_zyl1_ist = pd.Series(np.repeat(T_zyl1_ist,len(df)))
            df=df.assign(T_zyl1_ist = T_zyl1_ist.values)     
            
            T_zyl2_ist = file[cycle]['T802I_Value']['block0_values'][:]
            T_zyl2_ist = pd.Series(np.repeat(T_zyl2_ist,len(df)))
            df=df.assign(T_zyl2_ist = T_zyl2_ist.values)   
            
            T_zyl3_ist = file[cycle]['T803I_Value']['block0_values'][:]
            T_zyl3_ist = pd.Series(np.repeat(T_zyl3_ist,len(df)))
            df=df.assign(T_zyl3_ist = T_zyl3_ist.values)   
            
            T_zyl4_ist = file[cycle]['T804I_Value']['block0_values'][:]
            T_zyl4_ist = pd.Series(np.repeat(T_zyl4_ist,len(df)))
            df=df.assign(T_zyl4_ist = T_zyl4_ist.values)   
            
            T_zyl5_ist = file[cycle]['T805I_Value']['block0_values'][:]
            T_zyl5_ist = pd.Series(np.repeat(T_zyl5_ist,len(df)))
            df=df.assign(T_zyl5_ist = T_zyl5_ist.values)   
            
            V_um_ist = file[cycle]['V4065_Value']['block0_values'][:]
            df['V_um_ist']=np.nan
            df.loc[0]['V_um_ist'] = V_um_ist
    
            p_um_ist = file[cycle]['p4072_Value']['block0_values'][:]
            df['p_um_ist']=np.nan
            df.loc[0]['p_um_ist'] = p_um_ist

            p_inj_max_ist = file[cycle]['p4055_Value']['block0_values'][:]
            df['p_inj_max_ist']=np.nan
            df.loc[0]['p_inj_max_ist'] = p_inj_max_ist
      
            t_dos_ist = file[cycle]['t4015_Value']['block0_values'][:]
            df['t_dos_ist']=np.nan
            df.loc[0]['t_dos_ist'] = t_dos_ist
            
            t_inj_ist = file[cycle]['t4018_Value']['block0_values'][:]
            df['t_inj_ist']=np.nan
            df.loc[0]['t_inj_ist'] = t_inj_ist

            t_press1_soll = file[cycle]['t312_Value']['block0_values'][:]
            df['t_press1_soll']=np.nan
            df.loc[0]['t_press1_soll'] = t_press1_soll

            t_press2_soll = file[cycle]['t313_Value']['block0_values'][:]
            df['t_press2_soll']=np.nan
            df.loc[0]['t_press2_soll'] = t_press2_soll            
            
            cycle_num = file[cycle]['f071_Value']['block0_values'][0,0]
            df['cycle_num']=np.nan
            df.loc[0]['cycle_num'] = cycle_num
            
            pkl.dump(df,open(save_path+'cycle'+str(cycle_num)+'.pkl','wb'))
        except:
            continue
    
    return None
        

def add_csv_to_pd_dataframe(df_file_path,csv_file_path):
    
    #Read df
    df = pkl.load(open(df_file_path,'rb'))
    
    cycle_num = df.loc[0]['cycle_num']
    
    #Read csv
    df_csv = pd.read_csv(csv_file_path,sep=';',index_col=0)

    # add measurements from csv to pd dataframe
    for key in df_csv.keys():
        df[key]=np.nan
        df.loc[0][key] = df_csv.loc[cycle_num][key]
        
    # some measurements are constant trajectories    
    df['Werkzeugtemperatur'] = df.loc[0]['Werkzeugtemperatur']
    df['Düsentemperatur'] = df.loc[0]['Düsentemperatur']
    df['Einspritzgeschwindigkeit'] = df.loc[0]['Einspritzgeschwindigkeit']
    
    # and need to be renamed    
    df.rename(columns = {'Werkzeugtemperatur':'T_wkz_soll',
                         'Düsentemperatur':'T_nozz_soll',
                         'Einspritzgeschwindigkeit':'v_inj_soll'}, inplace = True)
        
    pkl.dump(df,open(df_file_path,'wb'))
    
    return df
    

def arrange_data_for_ident(df,x,u_inj,u_press,u_cool):
    
    # Find switching instances
    
    # Assumption: First switch is were pressure is maximal
    t_um1 = df['p_inj_ist'].idxmax()
    idx_t_um1 = np.argmin(abs(df.index.values-t_um1))
    
    # Second switch results from 
    t_um2 = t_um1 + df.loc[0]['t_press1_soll'] + df.loc[0]['t_press2_soll']
    
    # find index closest to calculated switching instances
    idx_t_um2 = np.argmin(abs(df.index.values-t_um2))
    t_um2 = df.index.values[idx_t_um2]
   
    inject = {}
    press = {}
    cool = {}
    
    inject['u'] = df.loc[0:t_um1][u_inj].values[0:-1,:]
    inject['x'] = df.loc[0:t_um1][x].values[1::,:]
    inject['x_init'] = df.loc[0][x].values
    
    press['u'] = df.loc[t_um1:t_um2][u_press].values[0:-1,:]
    press['x'] = df.loc[t_um1:t_um2][x].values[1::,:]
    press['x_init'] = df.loc[t_um1][x].values

    cool['u'] = df.loc[t_um2::][u_cool].values[0:-1,:]
    cool['x'] = df.loc[t_um2::][x].values[1::,:]
    cool['x_init'] = df.loc[t_um2][x].values    
    
    return inject,press,cool


def eliminate_outliers(doe_plan):
    
    # eliminate all nan
    data = data[data.loc[:,'Gewicht']>=5]
    data = data[data.loc[:,'Stegbreite_Gelenk']>=4]
    data = data[data.loc[:,'Breite_Lasche']>=4]
    
    
    
    
    
    
    