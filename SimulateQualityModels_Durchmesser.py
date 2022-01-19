# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:45:55 2021

@author: alexa
"""
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



dim_c = 2
path = 'Results/17_01_2022/'

results_train = pkl.load(open(path+'results_train_c13.pkl','rb')) 
results_val = pkl.load(open(path+'results_val_c13.pkl','rb')) 
quality_model = pkl.load(open(path+'quality_model_c13.pkl','rb'))
data= pkl.load(open(path+'data_c13.pkl','rb'))

idx_best = results_val['e'].idxmin()




idx_591 = np.where(results_val.index == 591)[0][0]
c_591,y_591 = quality_model.Simulation(data['init_state_val'][idx_591], 
                                         data['u_val'][idx_591],None,
                                         data['switch_val'][idx_591])

idx_431 = np.where(results_val.index == 431)[0][0]
c_431,y_431 = quality_model.Simulation(data['init_state_val'][idx_431], 
                                         data['u_val'][idx_431],None,
                                         data['switch_val'][idx_431])

idx_161 = np.where(results_val.index == 161)[0][0]
c_161,y_161 = quality_model.Simulation(data['init_state_val'][idx_161], 
                                         data['u_val'][idx_161],None,
                                         data['switch_val'][idx_161])



fig,ax = plt.subplots()
ax.plot(c_591[:,0],'k')
ax.plot(c_431[:,0],'m')
ax.plot(c_161[:,0],'c')
ax.legend(['591','431','161'])
ax.set_xlabel('k')

ax2=ax.twinx()
ax2.plot(c_591[:,1],'k')
ax2.plot(c_431[:,1],'m')
ax2.plot(c_161[:,1],'c')

ax2.plot(y_591.shape[0],y_591[-1],'ko')
ax2.plot(y_431.shape[0],y_431[-1],'mo')
ax2.plot(y_161.shape[0],y_161[-1],'co')
ax2.set_ylim([-2.5,2.5])




plt.figure()
sns.stripplot(x=results_train.index, y="y_true", data=results_train,
              size=4, color=".3", linewidth=0)
sns.stripplot(x=results_train.index, y="y_est", data=results_train,
              size=4,  linewidth=0)

plt.figure()
sns.stripplot(x=results_val.index, y="y_true", data=results_val,
              size=4, color=".3", linewidth=0)
sns.stripplot(x=results_val.index, y="y_est", data=results_val,
              size=4,  linewidth=0)

plt.figure()
plt.plot(np.vstack(data['u_val'][idx_591])[:,0:2],'k')
plt.plot(np.vstack(data['u_val'][idx_431])[:,0:2],'m')
plt.plot(np.vstack(data['u_val'][idx_161])[:,0:2],'c')


plt.figure()
plt.plot(results_train['y_true'],results_train['e'],'ko')
plt.plot(results_val['y_true'],results_val['e'],'mo')
plt.legend(['train','val'])
plt.xlabel('y_true')
plt.ylabel('e')

plt.figure()
plt.plot(results_train['y_true'],results_train['y_est'],'ko')
plt.plot(results_val['y_true'],results_val['y_est'],'mo')
plt.legend(['train','val'])
plt.xlabel('y_true')
plt.ylabel('y_est')
# plt.figure()
# plt.hist(results_train['e'].values,bins=40)
# plt.xlabel(['error'])

# plt.figure()
# plt.plot(results_train['cycle'],results_train['y_true'],'o')
# plt.plot(results_val['cycle'],results_val['y_true'],'o')


# plt.plot(np.array(data['y_val']),np.array(y_val),'o')
# plt.xlim([27.2,27.9])
# plt.ylim([27.2,27.9])

# plt.figure()
# plt.hist(np.array(e_val),bins=40)
# plt.xlabel(['error'])

# plt.figure()
# plt.plot(np.array(data['y_val']),np.array(e_val),'o')
# plt.xlabel(['y_true'])
# plt.ylabel(['error'])


# plt.figure()
# sns.stripplot(x="charge", y="e", data=results_train,
#               size=4, color=".3", linewidth=0)
# sns.stripplot(x="charge", y="e", data=results_val,
#               size=4,  linewidth=0)



# # Charge 1
# idx_train = cycles_train_label[np.where(charge_train_label[:] == 1)[0]].reshape((-1,))
# idx_val = cycles_val_label[np.where(charge_val_label == 1)[0]].reshape((-1,))

# plt.figure()
# plt.plot(results_train[results_train['charge']==1]['y_true'],'d',markersize=12,label='train true')
# plt.plot(results_train[results_train['charge']==1]['y_est'],'d',markersize=12,label='train est')
# # plt.ylim([27.5,27.9])

# plt.figure()
# plt.plot(results_train[results_train['charge']==1]['y_true'],'d',markersize=12,label='val true')
# plt.plot(results_train[results_train['charge']==1]['y_est'],'d',markersize=12,label='val est')
# plt.ylim([27.5,27.9])



# plt.subplot(2,3,6,title='D innen',xlabel='cycle')
# plt.plot([2,3,4,5,6],q_train,'d',markersize=12,label='train true')
# plt.plot([7,8,9],q_val,'d',markersize=12,label='val true')
# plt.legend()
# plt.subplots_adjust(hspace=0.3)
# plt.show()

'''
TO DO:
- Residuen gruppiert nach Faktorstufen plotten
- Residuen im Histogramm plotten
- Residuen über wahre Zielgröße plotten    
- Woher kommen harte Begrenzungen oben und unten?
'''


# e_val = np.array(e_val).reshape((-1,1))
# plt.hist(e_val)



# u_lab= ['p_wkz_ist','T_wkz_ist']#,'p_inj_ist','Q_Vol_ist','V_Screw_ist']
# plt.close('all')

# plt.figure()

# for i in range(0,1):
    
#     cycle_num = cycles_val_label[i]

#     cycle_data = pkl.load(open('./data/Versuchsplan/cycle'+str(cycle_num)+'.pkl','rb'))
    
#     # plt.figure()
#     # plt.plot(cycle_data[u_lab])
#     # plt.legend(u_lab)
#     # plt.title(str(cycle_num))
    
#     plt.figure()
#     plt.plot(cycle_data.index[0:y_val_hist[i].shape[0]],y_val_hist[i])
#     plt.legend(['Q'])
#     plt.title(str(cycle_num))

#     plt.figure()
#     plt.plot(cycle_data.index[0:c_val_hist[i].shape[0]],c_val_hist[i])
#     plt.legend(['c1','c2','c3','c4','c5','c6','c7'])
#     plt.title(str(cycle_num))
    
    # y_lab = ['Durchmesser_innen']
    # u_inj_lab= ['p_wkz_ist','T_wkz_ist','p_inj_ist','Q_Vol_ist','V_Screw_ist']
    # u_press_lab = u_inj_lab
    # u_cool_lab = ['p_wkz_ist','T_wkz_ist']

















