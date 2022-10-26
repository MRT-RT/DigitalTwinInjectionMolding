# -*- coding: utf-8 -*-

  
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pickle as pkl
from scipy import stats
import h5py
import time


class PIM_Data():
    
    def __init__(self,source_hdf5, target_hdf5,charts,scalar,scalar_dtype,
                 features,features_dtype,quals,quals_dtype,setpoints):
        
        self._source_hdf5 = source_hdf5
        self._target_hdf5 = target_hdf5
        
        self.charts = charts
        self.scalar = scalar
        self.features = features
        self.quals = quals
        
        self.scalar_dtype = scalar_dtype
        self.features_dtype = features_dtype
        self.quals_dtype = quals_dtype
        
        self.setpoints = setpoints
        
        self.target_hdf5 = target_hdf5
        
        self.failed_cycles = []

    @property
    def source_hdf5(self):
        return self._source_hdf5
    
    @source_hdf5.setter
    def source_hdf5(self,source_hdf5):
        self._source_hdf5 = source_hdf5
    
    @property
    def target_hdf5(self):
        return self._target_hdf5
    
    @target_hdf5.setter
    def target_hdf5(self,target_hdf5):
        
        # Check if file exists
        if not (target_hdf5.exists() and target_hdf5.is_file()):
            
            # Create target hdf5 file
            target_file = h5py.File(target_hdf5,'w')
            target_file.create_group('overview')
            target_file.create_group('process_values')
            target_file.create_group('features') 
            target_file.create_group('quality_meas')
            target_file.close()
            
            df_scalar = pd.DataFrame(data = [],
                             columns = [self.scalar[key] for key in self.scalar.keys()])
            df_scalar = df_scalar.set_index('Zyklus')
            df_scalar.to_hdf(target_hdf5,'overview')
    
            df_feat = pd.DataFrame(data = [],columns = self.features)
            df_feat.index = df_scalar.index
            df_feat.to_hdf(target_hdf5,'features')
            
            df_qual = pd.DataFrame(data=[],columns=self.quals)
            df_qual.index.rename('Zyklus',inplace=True)
            df_qual.to_hdf(target_hdf5,'quality_meas') 
            
            df_modelling = pd.concat([df_scalar,df_feat,df_qual],axis=1)
            df_modelling.to_hdf(target_hdf5,'modelling_data') 
            
        self._target_hdf5 = target_hdf5

        
    def get_cycle_data(self):
        
        new_data = False
        
        with h5py.File(self.target_hdf5,mode='r') as target_file:
            read_cycles = set(target_file['process_values'].keys())
                
        with h5py.File(self.source_hdf5, 'r') as file:
            source_cycles = list(file.keys())
            # source_cycles = ['cycle_119','cycle_120','cycle_122']
        
        new_source_cycles = set(source_cycles)-read_cycles
        new_source_cycles = list(new_source_cycles - set(self.failed_cycles))
        
        if new_source_cycles:
            
            new_data = True
            
            charts = {}
            scalars = []
            features = []
            quals = []
            
            for cycle in new_source_cycles:                                # [0:10] for debugging
                
                print(cycle)
                
                try:
                    
                    df_scalar = self.read_scalars(cycle)
                    
                    cycle_number = int(df_scalar.index[0])
                                       
                    # read monitoring charts
                    df_chart = self.read_charts(cycle)
                    
                    # read setpoints
                    df_feat = self.calc_features(df_chart,cycle_number)

                    # read quality data
                    df_qual = self.read_quality(cycle,cycle_number)

                except:
                    # Remember failed cycles, does not include double cycles!
                    self.failed_cycles.append(cycle)
                    
                    continue
                
                charts[cycle_number] = df_chart
                scalars.append(df_scalar)                        
                features.append(df_feat)                        
                quals.append(df_qual)
                                
            # Concatenate list to pd.DataFrame
            df_scalar = pd.concat(scalars)
            df_feat = pd.concat(features)
            df_qual = pd.concat(quals)
            
            # If duplicates exist keep none
            double_idx = df_scalar.index[df_scalar.index.duplicated(keep=False)]
            unique_idx = df_scalar.index[~df_scalar.index.duplicated(keep=False)]
            
            if unique_idx.empty:
                return False
            
            # Keep only unique cycle data
            df_scalar = df_scalar.loc[unique_idx]
            df_feat = df_feat.loc[unique_idx]
            df_qual = df_qual.loc[unique_idx]  
            [charts.pop(idx,None) for idx in double_idx] 
            
            # Update data used for modelling
            df_modelling = pd.concat([df_scalar,df_feat,df_qual],axis=1)
            self.update_modelling_data(df_modelling)
            
            
            # Load data saved in hdf5
            df_scalar_old = pd.read_hdf(self.target_hdf5, 'overview')
            df_feat_old = pd.read_hdf(self.target_hdf5, 'features')
            df_qual_old = pd.read_hdf(self.target_hdf5, 'quality_meas')
            
            # Concat new and old data
            df_scalar = pd.concat([df_scalar_old,df_scalar])
            df_feat = pd.concat([df_feat_old,df_feat])
            df_qual = pd.concat([df_qual_old,df_qual])
            
            
            # recast because pandas is stupid
            for col in df_scalar.columns:
                df_scalar[col] = df_scalar[col].astype(self.scalar_dtype[col])
            
            for col in df_feat.columns:
                df_feat[col] = df_feat[col].astype(self.features_dtype[col])
            
            for col in df_qual.columns:
                df_qual[col] = df_qual[col].astype(self.quals_dtype[col])
             
                
            try: 
                # Save concatenated data
                df_scalar.to_hdf(self.target_hdf5,'overview')
                df_feat.to_hdf(self.target_hdf5,'features')
                df_qual.to_hdf(self.target_hdf5,'quality_meas')
                
                
                for key,value in charts.items():
                    charts[key].to_hdf(self.target_hdf5, 'process_values/cycle_'+str(key))
            except:
                print('Error during writing.')
                
            
                
        return new_data
  
    def read_charts(self,cycle_key):
        
        charts = [] 
        
        for chart in self.charts:
            
            data = [pd.read_hdf(self.source_hdf5,cycle_key+'/'+key) for
                    key in chart['keys']]
            
            data = [d.set_index('time') for d in data]
            
            # Rename columns
            for d in data:
                d.columns = columns=chart['values']
            
            # Concatenate data from different charts
            data = pd.concat(data,axis=0)
                        
            charts.append(data)
            
        charts = pd.concat(charts,axis=1)
            
        return charts
        
        
    def read_scalars(self,cycle_key):
        
        scalars = []
        
        for key,value in self.scalar.items():
            
            try:
                scalar_value = pd.read_hdf(self.source_hdf5,cycle_key+'/'+key)
                scalar_value.columns = [value]
            except:
                scalar_value = pd.DataFrame(data=None,columns=[value],
                                              index=[0])
                print(key + ' could not be read from file')
            
            scalars.append(scalar_value)
            
        df_scalar = pd.concat(scalars,axis=1)
        df_scalar = df_scalar.set_index('Zyklus')
        
        for col in df_scalar.columns:
            df_scalar[col] = df_scalar[col].astype(self.scalar_dtype[col])

        
        return df_scalar

    def read_quality(self,cycle_key,cycle_number):
        
        df_qual = pd.read_hdf(self.source_hdf5,cycle_key+'/add_data')
        df_qual.index = [cycle_number]

        for col in df_qual.columns:
            df_qual[col] = df_qual[col].astype(self.quals_dtype[col])
        
        return df_qual
    
    def calc_features(self,df_chart,cycle_number):
        
        T_wkz_0 = df_chart.loc[0,'T_wkz_ist']
        
        df_feat = pd.DataFrame(data=[T_wkz_0],columns = self.features,
                               index = [cycle_number])

        for col in df_feat.columns:
            df_feat[col] = df_feat[col].astype(self.features_dtype[col])

        return df_feat      

    def update_modelling_data(self,df_new):
        
        # Sort after cycle number (time would be better but format is messed up)
        df_new = df_new.sort_index()
        
        # just for debugging
        # df_new.loc[427,self.setpoints[0]]=15.0
        self.n_max = 20
        
        # Load old modelling data
        df_mod_old = pd.read_hdf(self.target_hdf5,'modelling_data')
        
        # in case it's the first run over the data, 'modelling_data' will be
        # empty. Create DataFrame with desired structure
        if df_mod_old.empty:
            
            # find unique setpoint combinations and label them by numbers
            df_unique = df_new.drop_duplicates(subset=self.setpoints)
            df_unique = df_unique[self.setpoints]
            
            # number of setpoints equals number of unique rows
            num_setpts = list(range(len(df_unique)))
            
            # Add a column for setpoints number label
            df_new['Setpoint'] = None 
            
            # index of rows that are kept for modelling
            idx_mod = []
            
            for s in num_setpts:
                # find rows belonging to the same setpoint and label them
                set_idx = (df_new[self.setpoints] == df_unique.iloc[s]).all(axis=1)
                df_new.loc[set_idx,'Setpoint'] = s
                
                if len(set_idx)>self.n_max:
                    # If more than the maximal number of observations per 
                    # setpoint exist, keep observations with largest temperatur 
                    # difference
                    
                    df_T0 = df_new.loc[df_new['Setpoint']==s,['T_wkz_0']]
                    df_T0 = df_T0.sort_values('T_wkz_0', ascending=True)
                    
                    # Keep only those n_max observations per setpoint that cover T_wkz_0
                    # best
                    df_T0 = df_T0.assign(diff=df_T0.diff())
                    
                    # sort by temperature diff, keep observations with largest
                    # difference between them
                    keep_idx = df_T0.iloc[1:-1].sort_values('diff',ascending=False).index
                    
                    keep_idx = [df_T0.index[0]] + list(keep_idx[0:self.n_max-2]) \
                        + [df_T0.index[-1]]
                    
                    idx_mod.extend(keep_idx)
                
                else:
                    # else keep all
                    idx_mod.extend(list(set_idx.index))
            
            # idx_mod = pd.concat(idx_mod,axis=0)
            
            df_mod = df_new.loc[idx_mod]
            
            df_mod['Setpoint'] = df_mod['Setpoint'].astype('int16')
            
            # save modelling data to hdf5
            df_mod.to_hdf(self.target_hdf5,'modelling_data')
        
        else:

            # Go through each row of new data
            for cyc in df_new.index:
                set_idx = (df_mod_old[self.setpoints] == df_new.loc[cyc,self.setpoints]).all(axis=1)    
                
                # Check if setpoint exist, if it doesn't append data
                if set_idx.empty:
                    new_setpt = int(df_mod_old['Setpoint'].max() + 1)
                    df_new.loc[cyc,'Setpoint'] = new_setpt
                    
                    df_mod = pd.concat([df_mod_old,df_new])
                    
                else:
                    #if setpoint data exist, check if more than maximum
                    setpt = df_mod_old.loc[set_idx,'Setpoint'].iloc[0]
                    df_new.loc[cyc,'Setpoint'] = int(setpt)
                    
                    if len(set_idx) >= self.n_max:
                        #if maximum is exceeded replace datum with closest temp
                        diff = abs(df_mod_old.loc[set_idx,'T_wkz_0']-\
                            df_new.loc[cyc,'T_wkz_0'])
                        del_row = diff.idxmin()
                        #delete datum with closest temp
                        df_mod_old = df_mod_old.drop(index=del_row,axis=1)
                    
                    df_mod = pd.concat([df_mod_old,df_new])

                    df_mod['Setpoint'] = df_mod['Setpoint'].astype('int16')
                    
                    df_mod.to_hdf(self.target_hdf5,'modelling_data')
            
            # replace of add observation for this setpoint
            # create new setpoint if it doesn't exist yet
        
        return None
        

def klemann_convert_hdf5(a,b):
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
            
            ''' Setpoints Anfang '''
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

            ''' Setpoints Ende ''' 
            
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
        u_all_lab = []
        
        for u in u_lab:
            u_all_lab.extend(u)
        u_all_lab = list(set(u_all_lab))
  
        # if len(u_lab)==1:
        #     u_all_lab = u_lab[0]
            
        # elif len(u_lab)==3:
        #     u_inj_lab = u_lab[0]
        #     u_press_lab = u_lab[1]
        #     u_cool_lab = u_lab[2]
            
        #     u_all_lab = u_inj_lab + list(set(u_press_lab) - set(u_inj_lab))
        #     u_all_lab = u_all_lab + list(set(u_cool_lab) - set(u_all_lab))
        # else:
        #     print('Either one or three subsystems are supported!')
        #     return None
            
        cycle = cycle[u_all_lab+y_lab]
        
        # Delete NaN and get outputs
        if mode == 'quality':
            nan_cycle = np.isnan(cycle[u_all_lab]).any(axis=1)
            cycle = cycle.loc[~nan_cycle]
            
            # y.append(cycle.loc[0,y_lab].values)
            
            # Read desired data from dataframe
            data.append(cycle)     
            x_init.append(None)
            switch.append([t1,t2])
            
            # if len(u_lab)==1:
            #     data.append(cycle)                
            #     x_init.append(None)
            #     switch.append([None])
                
            # elif len(u_lab)==3:
                   
            #     data.append(cycle)
            #     x_init.append(None)
            #     switch.append([t1,t2])
                
        
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
    
    y_lab = ['Gewicht','Breite_Lasche','Durchmesser_innen','E-Modul','Maximalspannung',
             'Stegbreite_Gelenk']
    y_filt = []
    for lab in y_lab:
        if lab in doe_plan.keys():
            y_filt.append(lab)
    
    # If all entries NaN then this measurement was not taken for all parts
    y_filt = [lab for lab in y_filt if not doe_plan[lab].isnull().values.all()]
    
    
    doe_plan_new = doe_plan[doe_plan[y_filt].notnull().all(axis=1)]
    
    doe_plan_no_out=  doe_plan_new[(np.abs(stats.zscore(doe_plan_new[y_filt])) < 3).all(axis=1)]
        
    return doe_plan_no_out

def split_charges_to_trainval_data(path,charges,split,del_outl):
    
	# Load Versuchsplan to find cycles that should be considered for modelling
	data = pkl.load(open(path+'/Versuchsplan.pkl','rb'))
    
	if del_outl is True:
		data = eliminate_outliers(data)
    
	# Delete outliers rudimentary
	# Cycles for parameter estimation
	cycles_train_label = []
	cycles_val_label = []
    
	charge_train_label = []
	charge_val_label = []

	for charge in charges:
		if charge==26:
			print('huhu')
		cycles = data[data['Charge']==charge].index.values
		
		remove_cycles = [767,764,753]
    
		for rem in remove_cycles:	
			try:
				cycles.remove(rem) 
			except:
				pass
        
		if split == 'part':
			cyc_t = list(set([*cycles[0:2],*cycles[-2:]]))
			cyc_v = list(set([cycles[2],cycles[-4]]))
            
		elif split == 'all':
			cyc_t = list(set([*cycles[0:2],*cycles[3:-4],*cycles[-3:]]))
			cyc_v = list(set([cycles[2],cycles[-4]]))
        
		elif split == 'process':
			cyc_t = list(set([*cycles[-3:-1]]))
			cyc_v = list(set([cycles[-1]]))

		elif split == 'random':
			cyc_v = list(np.random.choice(cycles,1,False))
			cyc_t = list(set(cycles)-set(cyc_v))

		elif split == 'inner':
			cyc_v = list(np.random.choice(cycles[1:-1],1,False))
			cyc_t = list(set(cycles)-set(cyc_v))
        
		cycles_train_label.extend(cyc_t)
		cycles_val_label.extend(cyc_v)
        
		charge_train_label.extend([charge]*len(cyc_t))
		charge_val_label.extend([charge]*len(cyc_v))
 
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

    
def LoadDynamicData(path,charges,split,y_lab,u_lab,mode,del_outl):
    
    cycles_train_label, charge_train_label, cycles_val_label, charge_val_label = \
    split_charges_to_trainval_data(path,charges,split,del_outl)
      
    # Load cycle data, check if usable, convert to numpy array
    cycles_train = []
    cycles_val = []
    
    for c in cycles_train_label:
        cycles_train.append(pkl.load(open(path+'/cycle'+str(c)+'.pkl',
                                          'rb')))
    
    for c in cycles_val_label:
        cycles_val.append(pkl.load(open(path+'/cycle'+str(c)+'.pkl',
                                          'rb')))      
        
            
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

def LoadFeatureData(path,charges, split,del_outl):
    
    cycles_train_label, charge_train_label, cycles_val_label, charge_val_label = \
    split_charges_to_trainval_data(path,charges,split,del_outl)    
       
    # load doe plan 
    doe_plan = pkl.load(open(path+'/Versuchsplan.pkl','rb'))

    data_train = doe_plan.loc[cycles_train_label]
    data_val = doe_plan.loc[cycles_val_label]
    
    data_train['charge'] = charge_train_label
    data_val['charge'] = charge_val_label
    
    # Load cycle data and extract features
    cycles_train = []
    cycles_val = []
    
    features=['T_wkz_0','T_wkz_max','t_Twkz_max','T_wkz_int','p_wkz_max',
              'p_wkz_int', 'p_wkz_res','t_pwkz_max','p_inj_int', 'p_inj_max',
              't_inj','x_inj','x_um','v_mean','p_wkz_0','p_inj_0','x_0']
    
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
            
            p_wkz_0 = cycle.loc[0]['p_wkz_ist']
            p_inj_0 = cycle.loc[0]['p_inj_ist']
            x_0 = cycle.loc[0]['V_Screw_ist']
            
            f = [T_wkz_0,T_wkz_max,t_Twkz_max,T_wkz_int,p_wkz_max,
                 p_wkz_int, p_wkz_res,t_pwkz_max,p_inj_int, p_inj_max, t_inj,
                 x_inj,x_um,v_mean,p_wkz_0,p_inj_0,x_0]
            
            # y = list(cycle.loc[0][targets].values)
            
            # f.extend(y)
            
            data.loc[c] = f
        # print(c)
    
    
    data_train = pd.concat([data_train,data_train_feat],axis=1)
    data_val = pd.concat([data_val,data_val_feat],axis=1)
    
    return data_train,data_val

def LoadSetpointData(path,charges, split,del_outl):
    
    cycles_train_label, charge_train_label, cycles_val_label, charge_val_label = \
    split_charges_to_trainval_data(path,charges,split,del_outl)    
        
    # Load cycle data and extract features
    cycles_train = []
    cycles_val = []
    
    doe_plan = pkl.load(open(path+'/Versuchsplan.pkl','rb'))
    
    # setpoints=list(doe_plan.columns[1:9])
    
    # setpoints.extend(targets)
    
    data_train = doe_plan.loc[cycles_train_label]#[setpoints]
    data_val = doe_plan.loc[cycles_val_label]#[setpoints]
    
    data_train['charge'] = charge_train_label
    data_val['charge'] = charge_val_label
    
    return data_train,data_val  


def MinMaxScale(df,**kwargs):
    
    df = df.copy()
    
    # Unnormalize data
    reverse = kwargs.pop('reverse',False)
     
    if reverse:
        
        col_min = kwargs['minmax'][0]
        col_max = kwargs['minmax'][1]
        
        if all(col_min.keys()==col_max.keys()):
            cols = col_min.keys()

        cols = [col for col in cols if col in df.columns]
        
        # Unscale from 0,1
        # df_rev = df[cols]* (col_max - col_min) + col_min
        
        # Unscale from -1,1
        df_rev = 1/2* ( (df[cols] + 1) * (col_max - col_min)) + col_min
        
        
        df.loc[:,cols] = df_rev
        
        return df
        
    # Normalize data
    else:
    
        try:
            col_min = kwargs['minmax'][0]
            col_max = kwargs['minmax'][1]
            
            if all(col_min.keys()==col_max.keys()):
                cols = col_min.keys()
            
            cols = [col for col in cols if col in df.columns]
            
        except:
            
            cols = kwargs['columns']
           
            col_min = df[cols].min()
            col_max = df[cols].max()
            
        # Scale to -1,1
        df_norm = 2*(df[cols] - col_min) / (col_max - col_min) - 1 
        
        df.loc[:,cols] = df_norm
        # Scale to 0,1   
        # df_norm = (df[columns] - col_min) / (col_max - col_min)
    
    
    
    return df,(col_min,col_max)

    