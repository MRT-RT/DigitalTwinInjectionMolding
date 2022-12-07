# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:49:50 2022

@author: alexa
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pickle as pkl
from scipy import stats
import h5py
import time

class Data_Manager():
    
    def __init__(self,source_hdf5, target_hdf5,charts,scalar,scalar_dtype,
                 features,features_dtype,quals,quals_dtype,setpoints):
        """
        

        Parameters
        ----------
        source_hdf5 : pathlib Path object
            path to the hdf5-file to which machine data is written, must exist.
        target_hdf5 : pathlib Path object
            path to the hdf5-file to which write the pre-processed data,
            is created if doesn't exist.
        charts : list of dictionaries        
            Each dictionary contains the keys 'keys' and 'values'. 'values'
            contains a list of strings. Each string is the user-chosen name of 
            the process values recorded in the charts in the right order. 
            'keys' contains a list with the hdf5-groupname of the charts in 
            source_hdf5 in chronological order.
            Example: [{'keys':['f3113I_Value','f3213I_Value','f3313I_Value'],
                       'values':['p_wkz_ist','p_hyd_ist','T_wkz_ist','p_hyd_soll',
                                 'state1']}]
        scalar : dictionary
            Dictionary with key:value pairs. key is a string containing the
            hdf5-groupname of the value that should be read from source_hdf5.
            value is a user specified name for that value.
            Example: {'T801I_Value':'T_zyl1_ist'}
        scalar_dtype : dictionary
            Dictionary with key:value pairs. key contains string of user-speci-
            fied name for that scalar value and value a string specifying the
            data type.
            Example: {'T_zyl1_ist':'float16'}
        features : list
            List of strings. Each string specifies a feature that should be 
            calculated from measured process values. That feature must be 
            defined in __calc_features().
            Example: ['T_wkz_0']
        features_dtype : dictionary
            Dictionary with key:value pairs. key contains string of user-speci-
            fied name for that feature value and value a string specifying the
            data type.
            Example: {'T_wkz_0':'float16'}
        quals : list
            List of strings. Each string specifies a quality measurement that 
            can be found in the group add_data in source_hdf5.
            Example: ['Durchmesser_innen', 'Durchmesser_außen']
        quals_dtype : dictionary
            Dictionary with key:value pairs. key contains string of user-speci-
            fied name for that quality value and value a string specifying the
            data type.
            Example: {'Durchmesser_innen':'float16',
                      'Durchmesser_außen':'float16'}
        setpoints : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
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
            
            print('Target file is created.')
            
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
        else:
            print('Target file already exists.')
            
        self._target_hdf5 = target_hdf5

        
    def get_cycle_data(self,**kwargs):
        
        
        delay = kwargs.pop('delay',0.0)
        num_cyc = kwargs.pop('num_cyc',None)
        update_mdata = kwargs.pop('update_mdata',True)
        
        new_data = False
        
        with h5py.File(self.target_hdf5,mode='r') as target_file:
            read_cycles = set(target_file['process_values'].keys())
        
        try:
            with h5py.File(self.source_hdf5, 'r') as file:
                source_cycles = list(file.keys())
        except OSError:
            print('Source hdf5 could not be opened, trying again.')
            return None
        
        new_source_cycles = set(source_cycles)-read_cycles
        new_source_cycles = list(new_source_cycles - set(self.failed_cycles))
        
        new_source_cycles.sort()
        
        if new_source_cycles:
            
            time.sleep(delay)
                        
            charts = {}
            scalars = []
            features = []
            quals = []
            
            for cycle in new_source_cycles[0:num_cyc]:
                
                print(cycle)
                if cycle == 'cycle_70009':
                    print('stop')
                
                try:
                    
                    df_scalar = self.__read_scalars(cycle)
                    
                    cycle_number = int(df_scalar.index[0])
                                       
                    # read monitoring charts
                    df_chart = self.__read_charts(cycle)
                    
                    # read setpoints
                    df_feat = self.__calc_features(df_chart,cycle_number)

                    # read quality data
                    df_qual = self.__read_quality(cycle,cycle_number)

                except:
                    # Remember failed cycles, does not include double cycles!
                    self.failed_cycles.append(cycle)
                    
                    continue
            
                charts[cycle_number] = df_chart
                scalars.append(df_scalar)                        
                features.append(df_feat)                        
                quals.append(df_qual)
                    
            
            if scalars:    
                # Concatenate list to pd.DataFrame
                df_scalar = pd.concat(scalars)
                df_feat = pd.concat(features)
                df_qual = pd.concat(quals)
                
            else:
                print('New data is not valid.')
                return False
            
            # If duplicates exist, delete them
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
            if update_mdata:
                df_modelling = pd.concat([df_scalar,df_feat,df_qual],axis=1)
                self.__update_modelling_data(df_modelling,update='cycle')
            
            
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
                    
                new_data = True
            except:
                print('Error during writing.')
                
        return new_data
  
    def __read_charts(self,cycle_key):
        
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
        
        
    def __read_scalars(self,cycle_key):
        
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

    def __read_quality(self,cycle_key,cycle_number):
        
        df_qual = pd.read_hdf(self.source_hdf5,cycle_key+'/add_data')
        df_qual.index = [cycle_number]

        for col in df_qual.columns:
            df_qual[col] = df_qual[col].astype(self.quals_dtype[col])
        
        return df_qual
    
    def __calc_features(self,df_chart,cycle_number):
        
        T_wkz_0 = df_chart.loc[0,'T_wkz_ist']
        
        df_feat = pd.DataFrame(data=[T_wkz_0],columns = self.features,
                               index = [cycle_number])

        for col in df_feat.columns:
            df_feat[col] = df_feat[col].astype(self.features_dtype[col])

        return df_feat      

    def __update_modelling_data(self,df_new,**kwargs):
        
        # unpack kwargs
        update = kwargs.pop('update','cycle')
        
        
        # Sort after cycle number (time would be better but format is messed up)
        df_new = df_new.sort_index()
        
        # just for debugging
        # df_new.loc[427,self.setpoints[0]]=15.0
        self.n_max = 20
        
        # Load old modelling data
        df_mod = pd.read_hdf(self.target_hdf5,'modelling_data')
        
        # in case it's the first run over the data, 'modelling_data' will be
        # empty. Create DataFrame with desired structure
        if df_mod.empty:
            
            # find unique setpoint combinations and label them by numbers
            df_unique = df_new.drop_duplicates(subset=self.setpoints)
            df_unique = df_unique[self.setpoints]
            
            # number of setpoints equals number of unique rows
            num_setpts = list(range(len(df_unique)))
            
            # Add a column for setpoints number label
            df_new['Setpoint'] = -1 
            
            # index of rows that are kept for modelling
            idx_mod = []
            
            for s in num_setpts:
                # find rows belonging to the same setpoint and label them
                set_idx = (df_new[self.setpoints] == df_unique.iloc[s]).all(axis=1)
                df_new.loc[set_idx,'Setpoint'] = s
                
                if len(df_new.loc[set_idx])>self.n_max:
                    # If more than the maximal number of observations per 
                    # setpoint exist, keep observations with largest temperatur 
                    # difference
                    
                    if update == 'T_wkz_0':
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
                            
                    elif update == 'cycle':
                        # Find index of most recent cycles
                        df_s = df_new.loc[df_new['Setpoint']==s].sort_index()
                        keep_idx = list(df_s.index[0:self.n_max])
                    
                    else:
                        raise ValueError('Choose an update method for modelling_data. Either "T_wkz_0" or "cycle".')
                    
                    
                    idx_mod.extend(keep_idx)
                
                else:
                    # else keep all
                    idx_mod.extend(list(df_new.loc[set_idx].index))
            
            # idx_mod = pd.concat(idx_mod,axis=0)
            
            df_mod = df_new.loc[idx_mod]
            
            df_mod['Setpoint'] = df_mod['Setpoint'].astype('int16')
            
            # save modelling data to hdf5
            df_mod.to_hdf(self.target_hdf5,'modelling_data')
        
        else:

            # Go through each row of new data
            for cyc in df_new.index:
                # Get boolean index of same setpoints as cyc 
                set_idx = (df_mod[self.setpoints] == df_new.loc[cyc,self.setpoints]).all(axis=1)    
                
                # Check if setpoint exist, if it doesn't append data
                if df_mod.loc[set_idx].empty:
                    new_setpt = int(df_mod['Setpoint'].max() + 1)
                    df_new.loc[cyc,'Setpoint'] = new_setpt
                    
                    df_mod = pd.concat([df_mod,df_new])
                    
                else:
                    #if setpoint data exist, check if more than maximum
                    setpt = df_mod.loc[set_idx,'Setpoint'].iloc[0]
                    df_new.loc[cyc,'Setpoint'] = int(setpt)
                    
                    if len(df_mod.loc[set_idx]) >= self.n_max:
                        #if maximum is exceeded replace datum according to update method
                        if update == 'T_wkz_0':
                            diff = abs(df_mod.loc[set_idx,'T_wkz_0']-\
                                df_new.loc[cyc,'T_wkz_0'])
                            del_row = diff.idxmin()                          
                            
                        elif update == 'cycle':
                            del_row = df_mod.loc[set_idx].index.min()

                        else:
                            raise ValueError('''Choose an update method for modelling_data. 
                                             Either "T_wkz_0" or "cycle".''')
                        
                        # Delete observation
                        df_mod = df_mod.drop(index=del_row,axis=1)
                    
                    # Add new observation
                    df_mod = pd.concat([df_mod,df_new.loc[[cyc]]])
                    
                    # Recast type
                    df_mod['Setpoint'] = df_mod['Setpoint'].astype('int16')
                    
                df_mod.to_hdf(self.target_hdf5,'modelling_data')
                   
        return None

    def get_machine_data(self):
        all_data = pd.read_hdf(self.target_hdf5, 'overview')
        return all_data

    def get_feature_data(self):
        feature_data = pd.read_hdf(self.target_hdf5, 'features')
        return feature_data

    def get_quality_data(self):
        quality_data = pd.read_hdf(self.target_hdf5, 'quality_meas')
        return quality_data

    def get_modelling_data(self):
        modelling_data = pd.read_hdf(self.target_hdf5, 'modelling_data')
        return modelling_data    

    def get_process_data(self,cycle):
        
        if type(cycle) == type(list()):
            cycle_data = []
            for c in cycle:
                cycle_data.append(pd.read_hdf(self.target_hdf5,
                                              'process_values/cycle_'+str(c)))
        else:
            cycle_data = pd.read_hdf(self.target_hdf5,
                                          'process_values/cycle_'+str(cycle))
            
        return cycle_data
                    
            
            
        
