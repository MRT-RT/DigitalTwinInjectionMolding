# -*- coding: utf-8 -*-

import DIM


daten = load(data)


num_neurons = [5,10,15]
num_states = [1,2,4,10]

for neurons,states in zip(num_neurons,num_states):
    
    
    model = DIM.models.GRU(neurons,states)
    
    results = DIM.optim.param_optim(data,model)


    
    
    