# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 09:33:19 2021

@author: alexa
"""

import pandas as pd
import h5py

filename = '20210825.h5'

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
    
# pd.read_hdf('20210825.h5','key')