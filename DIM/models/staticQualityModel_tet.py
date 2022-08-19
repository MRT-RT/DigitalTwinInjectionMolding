# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:33:50 2022

@author: LocalAdmin
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import pickle as pkl

import sys


path_dim = Path.cwd().parents[1]
sys.path.insert(0, path_dim.as_posix())

from DIM.miscellaneous.PreProcessing import LoadFeatureData
from functions import estimate_polynomial
from DIM.models.model_structures import DoubleExponential
from DIM.models.injection_molding import staticQualityModel
from DIM.optim.param_optim import ModelTraining, static_mode