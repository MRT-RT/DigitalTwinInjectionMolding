# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 10:38:04 2022

@author: alexa
"""

import sys
sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")

import h5py  
import time

from DIM.miscellaneous.PreProcessing import klemann_convert_hdf5

# Open hdf5-file
# Check if a new key/group has appeared
# if False close and open again a few seconds later
# if True open new group and parse data to new hdf5 file in new format

# path to hdf5 file with raw data created by PIM machine
hdf5_path = \
'Z:/Versuchsreihen Spritzgießen/Versuchsplan/Prozessgrößen_20211005.h5' 

# path to new hdf5 file with converted data
hdf5_path_new = 'test_folder/data.h5'

# for now we assume that the cycle counter is increased by 1, in practice 
# the counter might skip a number of be resetted, we'll deal with these exceptions
# in future

# Initialize cycle counter
c = 1000

execute_program = True

while execute_program:
    
    time.sleep(2.0)
    
    # open file
    file = h5py.File(hdf5_path,'r')
    
    # check for a new cycle
    try:
        cycle = file1['cycle_'+str(c)]
    
    # if cycle doesn't exist yet
    except KeyError:
        
        print('Key' 'cycle_'+str(c) + " doesn't exist")
        # go to next iteration of loop
        continue
    
    # Pass handle to new cycle to your function
    # The function should convert the data, write it into the new hdf5 file
    # and close the new hdf5 file
    
    # create handle on hdf5-file with converted data
    new_hdf5 = h5py.File(hdf5_path_new,'a')
    
    klemann_convert_hdf5(cycle,new_hdf5)
    
    new_hdf5.close()
    file.close()
    
    
        
