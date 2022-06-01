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
    
    success = []
    
    # print(file)
    for cycle in file.keys():
        
        try:
            # monitoring charts 1-3
            f3103I = file[cycle]['f3103I_Value']['block0_values'][:,[0,1]]
            f3203I = file[cycle]['f3203I_Value']['block0_values'][:,[0,1]]
            f3303I = file[cycle]['f3303I_Value']['block0_values'][:,[0,1]]    
            timestamp1 = np.vstack([f3103I[:,[0]],f3203I[:,[0]],f3303I[:,[0]]])
                         
            # measuring chart
            f3113I = file[cycle]['f3113I_Value']['block0_values'][:,0:5]
            f3213I = file[cycle]['f3213I_Value']['block0_values'][:,0:5]
            f3313I = file[cycle]['f3313I_Value']['block0_values'][:,0:5]
            timestamp2 = np.vstack([f3113I[:,[0]],f3213I[:,[0]],f3313I[:,[0]]])  
            
            # monitoring charts 4-6
            f3403I = file[cycle]['f3403I_Value']['block0_values'][:,[0,1]]
            f3503I = file[cycle]['f3503I_Value']['block0_values'][:,[0,1]]
            f3603I = file[cycle]['f3603I_Value']['block0_values'][:,[0,1]]  
            timestamp3 = np.vstack([f3403I[:,[0]],f3503I[:,[0]],f3603I[:,[0]]])
            
            MonChart1_3 = np.vstack((f3103I,f3203I,f3303I)) 
            MeasChart1_3 = np.vstack((f3113I,f3213I,f3313I))
            MonChart4_6 = np.vstack((f3403I,f3503I,f3603I))

                         
            df1 = pd.DataFrame(data=MonChart1_3[:,[1]],
            index = timestamp1[:,0], columns = ['Q_Vol_ist'])

            df2 = pd.DataFrame(data=MeasChart1_3[:,1:5],
            index = timestamp2[:,0], columns = ['p_wkz_ist', 'T_wkz_ist', 'p_inj_soll','p_inj_ist'])

            df3 = pd.DataFrame(data=MonChart4_6[:,[1]],
            index = timestamp3[:,0], columns = ['V_Screw_ist'])
    

            
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

            V_um_soll = file[cycle]['V305_Value']['block0_values'][:]
            df['Umschaltpunkt']=np.nan
            df.loc[0]['Umschaltpunkt'] = V_um_soll
            
            V_um_ist = file[cycle]['V4065_Value']['block0_values'][:]
            df['V_um_ist']=np.nan
            df.loc[0]['V_um_ist'] = V_um_ist
    
            p_um_ist = file[cycle]['p4072_Value']['block0_values'][:]
            df['p_um_ist']=np.nan
            df.loc[0]['p_um_ist'] = p_um_ist

            p_inj_max_ist = file[cycle]['p4055_Value']['block0_values'][:]
            df['p_inj_max_ist']=np.nan
            df.loc[0]['p_inj_max_ist'] = p_inj_max_ist
            
            p_press_soll = file[cycle]['p312_Value']['block0_values'][:]
            df['Nachdruckhöhe']=np.nan
            df.loc[0]['Nachdruckhöhe'] = p_press_soll   
      
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
            
            
            p_stau = file[cycle]['p403_Value']['block0_values'][:]
            df['Staudruck']=np.nan
            df.loc[0]['Staudruck'] = p_stau    
            
            df=df.assign(Einspritzgeschwindigkeit = Q_inj_soll.values)
            df=df.assign(Nachdruckzeit = t_press1_soll.item())
            
            pkl.dump(df,open(save_path+'cycle'+str(cycle_num)+'.pkl','wb'))
            
            success.append(cycle_num)
        except:
            continue
    
    return success


def hdf5_to_pd_dataframe_high_freq(file,save_path=None):
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
            
            # monitoring charts 1-3
            f3113I = file[cycle]['f3113I_Value']['block0_values'][:,[0,1,2,3,4]]
            f3213I = file[cycle]['f3213I_Value']['block0_values'][:,[0,1,2,3,4]]  
            f3313I = file[cycle]['f3313I_Value']['block0_values'][:,[0,1,2,3,4]] 
            
            # monitoring charts 1-3
            f3403I = file[cycle]['f3403I_Value']['block0_values'][:,[0,1]]  
            f3503I = file[cycle]['f3503I_Value']['block0_values'][:,[0,1]]  
            f3603I = file[cycle]['f3603I_Value']['block0_values'][:,[0,1]] 
            
            timestamp1 = np.vstack([f3103I[:,[0]],f3203I[:,[0]],f3303I[:,[0]]])
            timestamp2 = np.vstack([f3113I[:,[0]],f3213I[:,[0]],f3313I[:,[0]]])
            timestamp3 = np.vstack([f3403I[:,[0]],f3503I[:,[0]],f3603I[:,[0]]])
            
            MonChart1_3 = np.vstack((f3103I,f3203I,f3303I))
            MeasChart1_3 = np.vstack((f3113I,f3213I,f3313I))
            MonChart4_6 = np.vstack((f3403I,f3503I,f3603I))
                         
                       
            df1 = pd.DataFrame(data=MonChart1_3[:,[1]],
            index = timestamp1[:,0], columns = ['Q_Vol_ist'])
            
            df2 = pd.DataFrame(data=MeasChart1_3[:,[1,2,3,4]],
            index = timestamp2[:,0], columns = ['p_wkz_ist','T_wkz_ist','p_inj_soll',
                                                'p_inj_ist'])
            
            df3 = pd.DataFrame(data=MonChart4_6[:,[1]],
            index = timestamp3[:,0], columns = ['V_Screw_ist'])           
            
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
            
            V_um_soll = file[cycle]['V305_Value']['block0_values'][:]
            df['Umschaltpunkt']=np.nan
            df.loc[0]['Umschaltpunkt'] = V_um_soll
            
            V_um_ist = file[cycle]['V4065_Value']['block0_values'][:]
            df['V_um_ist']=np.nan
            df.loc[0]['V_um_ist'] = V_um_ist
    
            p_um_ist = file[cycle]['p4072_Value']['block0_values'][:]
            df['p_um_ist']=np.nan
            df.loc[0]['p_um_ist'] = p_um_ist

            p_inj_max_ist = file[cycle]['p4055_Value']['block0_values'][:]
            df['p_inj_max_ist']=np.nan
            df.loc[0]['p_inj_max_ist'] = p_inj_max_ist
            
            p_press_soll = file[cycle]['p312_Value']['block0_values'][:]
            df['Nachdruckhöhe']=np.nan
            df.loc[0]['Nachdruckhöhe'] = p_inj_max_ist            
      
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
        

def add_csv_to_pd_dataframe(df,df_csv):
    
    #Read df
    
    
    cycle_num = df.loc[0]['cycle_num']
    
    ###########################################################################
    
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
        
    # pkl.dump(df,open(df_file_path,'wb'))
    
    return df
    

# def arrange_data_for_qual_ident(cycles,x_lab,q_lab):
    
#     x = []
#     q = []
#     switch = []
    
#     for cycle in cycles:
#         # Find switching instances
#         # Assumption: First switch is were pressure is maximal
#         t1 = cycle['p_inj_ist'].idxmax()
#         idx_t1 = np.argmin(abs(cycle.index.values-t1))
        
#         # Second switch results from 
#         t2 = t1 + cycle.loc[0]['t_press1_soll'] + cycle.loc[0]['t_press2_soll']
        
#         # find index closest to calculated switching instances
#         idx_t2 = np.argmin(abs(cycle.index.values-t2))
#         t2 = cycle.index.values[idx_t2]
        
#         # Read desired data from dataframe
#         temp = cycle[x_lab].values                                              # can contain NaN at the end
#         nana = np.isnan(temp).any(axis=1)                                       # find NaN
        
#         x.append(temp[~nana,:])
#         q.append(cycle.loc[0,q_lab].values)
#         switch.append([idx_t1,idx_t2])
   
#     # inject = {}
#     # press = {}
#     # cool = {}
    
#     # inject['u'] = df.loc[0:t_um1][u_inj].values[0:-1,:]
#     # inject['x'] = df.loc[0:t_um1][x].values[1::,:]
#     # inject['x_init'] = df.loc[0][x].values
    
#     # press['u'] = df.loc[t_um1:t_um2][u_press].values[0:-1,:]
#     # press['x'] = df.loc[t_um1:t_um2][x].values[1::,:]
#     # press['x_init'] = df.loc[t_um1][x].values

#     # cool['u'] = df.loc[t_um2::][u_cool].values[0:-1,:]
#     # cool['x'] = df.loc[t_um2::][x].values[1::,:]
#     # cool['x_init'] = df.loc[t_um2][x].values    
    
#     return x,q,switch

def find_switches(cycle):
    '''
    

    Parameters
    ----------
    cycle : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    
    '''
    
    # Assumption: First switch is were pressure is maximal
    t1 = cycle['p_inj_ist'].idxmax()
    idx_t1 = cycle.index.get_loc(t1)
    
    # Second switch 
    t2 = t1 + cycle.loc[0]['t_press1_soll'] + cycle.loc[0]['t_press2_soll']
    idx_t2 = np.argmin(abs(cycle.index.values-t2))
    t2 = cycle.index[idx_t2]
    
    # Third switch, machine opens
    t3 = t2 + cycle.loc[0,'Kühlzeit']
    idx_t3 = np.argmin(abs(cycle.index.values-t3))
    t3 = cycle.index[idx_t3]
    
    return t1,t2,t3

def arrange_data_for_ident(cycles,y_lab,u_lab,mode):
    
    u = []
    x_init = []
    y = []
    data = []
    switch = []
    
    for cycle in cycles:
        
        # Find switching instances
        t1,t2,t3 = find_switches(cycle)
        
        # Reduce dataframe to desired variables and treat NaN
        if len(u_lab)==1:
            u_all_lab = u_lab[0]
            
        elif len(u_lab)==3:
            u_inj_lab = u_lab[0]
            u_press_lab = u_lab[1]
            u_cool_lab = u_lab[2]
            
            u_all_lab = u_inj_lab + list(set(u_press_lab) - set(u_inj_lab))
            u_all_lab = u_all_lab + list(set(u_cool_lab) - set(u_all_lab))
        else:
            print('Either one or three subsystems are supported!')
            return None
            
        cycle = cycle[u_all_lab+y_lab]
        
        # Delete NaN and get outputs
        if mode == 'quality':
            nan_cycle = np.isnan(cycle[u_all_lab]).any(axis=1)
            cycle = cycle.loc[~nan_cycle]
            
            # y.append(cycle.loc[0,y_lab].values)
            
            # Read desired data from dataframe
            if len(u_lab)==1:
                data.append(cycle)                
                x_init.append(None)
                switch.append([None])
                
            elif len(u_lab)==3:
                   
                data.append(cycle)
                x_init.append(None)
                switch.append([t1,t2])
                
        
        elif mode == 'process':
            
            nan_cycle = np.isnan(cycle[u_all_lab+y_lab]).any(axis=1)
            cycle = cycle.loc[~nan_cycle]
        
            # Read desired data from dataframe
            if len(u_lab)==1:
                u_all = cycle[u_all_lab].values
                u.append([u_all])
                switch.append([None])
                
            elif len(u_lab)==3:
                # In this version input and output data was shifted to each other
                # u_inj = cycle.loc[0:t1][u_inj_lab].values[0:-1]                         # can contain NaN at the end
                # u_press = cycle.loc[t1:t2][u_press_lab].values[0:-1]  
                # u_cool = cycle.loc[t2:t3][u_cool_lab].values[0:-1]  
    
                # y_inj = cycle.loc[0:t1][y_lab].values[1::]                              # can contain NaN at the end
                # y_press = cycle.loc[t1:t2][y_lab].values[1::]  
                # y_cool = cycle.loc[t2:t3][y_lab].values[1::]      
                
                
                # u_inj = cycle.loc[0:t1][u_inj_lab].values                                 # can contain NaN at the end
                # u_press = cycle.loc[t1:t2][u_press_lab].values  
                # u_cool = cycle.loc[t2:t3][u_cool_lab].values  
    
                # y_inj = cycle.loc[0:t1][y_lab].values                                     # can contain NaN at the end
                # y_press = cycle.loc[t1:t2][y_lab].values  
                # y_cool = cycle.loc[t2:t3][y_lab].values      
    
                # u.append([u_inj,u_press,u_cool])
                # y.append(np.vstack([y_inj,y_press,y_cool]))
                
                data.append(cycle)
                x_init.append(cycle.loc[0][y_lab].values.reshape(-1,1))
                switch.append([t1,t2])
                # switch.append([cycle.index.get_loc(t1),cycle.index.get_loc(t2)])
                
                
                
        
    return data,x_init,switch



def eliminate_outliers(doe_plan):
    
    # eliminate all nan
    doe_plan = doe_plan[doe_plan.loc[:,'Gewicht']>=5]
    # doe_plan = doe_plan[doe_plan.loc[:,'Stegbreite_Gelenk']>=4]
    doe_plan = doe_plan[doe_plan.loc[:,'Breite_Lasche']>=4]
    doe_plan = doe_plan[doe_plan.loc[:,'Durchmesser_innen']>=26.5]
    
    return doe_plan

def split_charges_to_trainval_data(path,charges,split):
    
    # Load Versuchsplan to find cycles that should be considered for modelling
    data = pkl.load(open(path+'Versuchsplan.pkl','rb'))
    
    data = eliminate_outliers(data)
    
    # Delete outliers rudimentary
    
    # Cycles for parameter estimation
    cycles_train_label = []
    cycles_val_label = []
    
    charge_train_label = []
    charge_val_label = []
    
    for charge in charges:
               
        cycles = data[data['Charge']==charge].index.values

        if split == 'part':
            cyc_t = list(set([*cycles[0:2],*cycles[-2:]]))
            cyc_v = list(set([cycles[2],cycles[-4]]))
            
        elif split == 'all':
            cyc_t = list(set([*cycles[0:2],*cycles[3:-4],*cycles[-3:]]))
            cyc_v = list(set([cycles[2],cycles[-4]]))
        
        elif split == 'process':
            cyc_t = list(set([*cycles[-3:-1]]))
            cyc_v = list(set([cycles[-1]]))

        cycles_train_label.extend(cyc_t)
        cycles_val_label.extend(cyc_v)
        
        charge_train_label.extend([charge]*len(cyc_t))
        charge_val_label.extend([charge]*len(cyc_v))

    # print(len(charge_train_label))
    # print(len(charge_val_label))
    
    cycles_train_label = np.hstack(cycles_train_label)
    cycles_val_label = np.hstack(cycles_val_label)
    # print(len(cycles_val_label))
    # Delete cycles that for some reason don't exist
    charge_train_label = np.delete(charge_train_label, np.where(cycles_train_label == 767)) 
    cycles_train_label = np.delete(cycles_train_label, np.where(cycles_train_label == 767)) 

    charge_train_label = np.delete(charge_train_label, np.where(cycles_train_label == 764)) 
    cycles_train_label = np.delete(cycles_train_label, np.where(cycles_train_label == 764)) 

    charge_train_label = np.delete(charge_train_label, np.where(cycles_train_label == 753)) 
    cycles_train_label = np.delete(cycles_train_label, np.where(cycles_train_label == 753))     
    
    return cycles_train_label, charge_train_label, cycles_val_label, charge_val_label

    
def LoadDynamicData(path,charges,split,y_lab,u_lab,mode,norm_cycle):
    
    cycles_train_label, charge_train_label, cycles_val_label, charge_val_label = \
    split_charges_to_trainval_data(path,charges,split)
      
    # Load cycle data, check if usable, convert to numpy array
    cycles_train = []
    cycles_val = []
    
    for c in cycles_train_label:
        cycles_train.append(pkl.load(open(path+'cycle'+str(c)+'.pkl',
                                          'rb')))
    
    for c in cycles_val_label:
        cycles_val.append(pkl.load(open(path+'cycle'+str(c)+'.pkl',
                                          'rb')))
    
    # Select input and output for dynamic model
    if len(u_lab)==3:
        u_inj_lab = u_lab[0]
        u_press_lab = u_lab[1]
        u_cool_lab = u_lab[2]
        
        u_lab_all = u_lab[0] + list(set(u_lab[1])-set(u_lab[0]))
        u_lab_all = u_lab_all + list(set(u_lab[2])-set(u_lab_all))
        
    elif len(u_lab)==1:
        u_lab_all = u_lab[0]
    else:
        raise ValueError('''u_lab needs to be a list of either one or three 
                         elements with input labels!''')

    # Normalize with respect to normalization cycle
    # mean_y = cycles_train[0][y_lab].mean() #norm_cycle[y_lab].mean()                                       # This normalization was formerly used for quality models
    
    # min_u = cycles_train[0][u_lab_all].min()#norm_cycle[u_lab_all].min()
    # max_u = cycles_train[0][u_lab_all].max()#norm_cycle[u_lab_all].max()
    
    # min_y = cycles_train[0][y_lab].min()#norm_cycle[y_lab].min()
    # max_y = cycles_train[0][y_lab].max()#norm_cycle[y_lab].max()

    # mean_y = norm_cycle[y_lab].mean()                                       # This normalization was formerly used for quality models
    
    # min_u = norm_cycle[u_lab_all].min()
    # max_u = norm_cycle[u_lab_all].max()
    
    # min_y = norm_cycle[y_lab].min()
    # max_y = norm_cycle[y_lab].max()

    # min_u[max_u-min_u==0]=0                                                     # if signal is constant, set minimum to 0 to avoid division by zero    
    # min_y[max_y-min_y==0]=0
    
    # print('p_inj_soll at switch point is set to 700 manually!')
    
    # for cycle in cycles_train+cycles_val:
        
    #     t1,_,_ = find_switches(cycle)
    #     cycle.loc[t1]['p_inj_soll']=700.00
        
        
    #     cycle[u_lab_all] = (cycle[u_lab_all]-min_u)/(max_u-min_u)
        
    #     if mode == 'quality':
    #         cycle[y_lab] = (cycle[y_lab]-mean_y)+1
    #     elif mode == 'process':
    #         cycle[y_lab] = (cycle[y_lab]-min_y)/(max_y-min_y)
        
        
            
    data_train,x_init_train,switch_train  = arrange_data_for_ident(cycles_train,y_lab,
                                        u_lab,mode)
    
    data_val,x_init_val,switch_val = arrange_data_for_ident(cycles_val,y_lab,
                                        u_lab,mode)
    
    data_train = {'data': data_train,
                'init_state': x_init_train,
                'switch': switch_train,
                'charge_num':charge_train_label,
                'cycle_num':cycles_train_label}
    
    data_val = {'data': data_val,
                'init_state': x_init_val,
                'switch': switch_val,
                'charge_num':charge_val_label,
                'cycle_num':cycles_val_label}
    
    return data_train,data_val

def LoadFeatureData(path,charges, split):
    
    cycles_train_label, charge_train_label, cycles_val_label, charge_val_label = \
    split_charges_to_trainval_data(path,charges,split)    
       
    # load doe plan 
    doe_plan = pkl.load(open(path+'Versuchsplan.pkl','rb'))

    data_train = doe_plan.loc[cycles_train_label]
    data_val = doe_plan.loc[cycles_val_label]
    
    data_train['charge'] = charge_train_label
    data_val['charge'] = charge_val_label
    
    # Load cycle data and extract features
    cycles_train = []
    cycles_val = []
    
    features=['T_wkz_0','T_wkz_max','t_Twkz_max','T_wkz_int','p_wkz_max',
              'p_wkz_int', 'p_wkz_res','t_pwkz_max','p_inj_int', 'p_inj_max',
              't_inj','x_inj','x_um','v_mean']
    
    # features.extend(targets)
    
    data_train_feat = pd.DataFrame(data=None,columns = features, index=cycles_train_label)
    data_val_feat = pd.DataFrame(data=None,columns = features, index=cycles_val_label)
    # print(cycles_val_label)
    # print(len(data_val))
       
    for data,cycle_labels in zip([data_train_feat,data_val_feat],[cycles_train_label,cycles_val_label]):
   
            
        for c in cycle_labels:
            
            # Load Data
            cycle = pkl.load(open(path+'/cycle'+str(c)+'.pkl','rb'))
            
            t1,t2,t3 = find_switches(cycle)
            
            cycle = cycle.loc[0:t3]                                                 # cut off data after tool opened
            # Extract features
            T_wkz_0 = cycle.loc[0]['T_wkz_ist']                                     # T at start of cycle
            T_wkz_max = cycle['T_wkz_ist'].max()                                    # max. T during cycle
            t_Twkz_max = cycle['T_wkz_ist'].idxmax()                                 # time when max occurs
            T_wkz_int = cycle['T_wkz_ist'].sum()                                    # T Integral
            
            p_wkz_max = cycle['p_wkz_ist'].max()                                    # max cavity pressure
            p_wkz_int = cycle['p_wkz_ist'].sum()                                    # integral cavity pressure
            p_wkz_res = cycle.loc[t2::]['p_wkz_ist'].mean()                         # so called pressure drop
            t_pwkz_max = cycle['p_wkz_ist'].idxmax()
            
            p_inj_int = cycle.loc[t1:t2]['p_inj_ist'].mean()                        # mean packing pressure
            p_inj_max = cycle['p_inj_ist'].max()                                    # max hydraulic pressure
            
            t_inj = t1                                                              # injection time
            
            x_inj = cycle.loc[0]['V_Screw_ist']-cycle.loc[t1]['V_Screw_ist']        # injection stroke
            x_um =  cycle.loc[t1]['V_Screw_ist']                                    # switch position
            v_mean = cycle.loc[0:t1]['Q_Vol_ist'].mean()                            # mean injection velocity
            
            f = [T_wkz_0,T_wkz_max,t_Twkz_max,T_wkz_int,p_wkz_max,
                 p_wkz_int, p_wkz_res,t_pwkz_max,p_inj_int, p_inj_max, t_inj,
                 x_inj,x_um,v_mean]
            
            # y = list(cycle.loc[0][targets].values)
            
            # f.extend(y)
            
            data.loc[c] = f
        # print(c)
    
    
    data_train = pd.concat([data_train,data_train_feat],axis=1)
    data_val = pd.concat([data_val,data_val_feat],axis=1)
    
    return data_train,data_val

def LoadSetpointData(path,charges, split):
    
    cycles_train_label, charge_train_label, cycles_val_label, charge_val_label = \
    split_charges_to_trainval_data(path,charges,split)    
        
    # Load cycle data and extract features
    cycles_train = []
    cycles_val = []
    
    doe_plan = pkl.load(open(path+'Versuchsplan.pkl','rb'))
    
    # setpoints=list(doe_plan.columns[1:9])
    
    # setpoints.extend(targets)
    
    data_train = doe_plan.loc[cycles_train_label]#[setpoints]
    data_val = doe_plan.loc[cycles_val_label]#[setpoints]
    
    data_train['charge'] = charge_train_label
    data_val['charge'] = charge_val_label
    
    return data_train,data_val  


def MinMaxScale(df,columns,*args):
    
    try:
        col_min = args[0]
        col_max = args[1]
    
    except:
        col_min = df[columns].min()
        col_max = df[columns].max()
        

    df_norm = 2*(df[columns] - col_min) / (col_max - col_min) - 1 
        
        
    return df_norm,(col_min,col_max)

    