# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:16:22 2022

@author: LocalAdmin
"""
from DIM.miscellaneous.PreProcessing import StaticFeatureExtraction

charges = [1]

data_train, data_val = StaticFeatureExtraction(charges)