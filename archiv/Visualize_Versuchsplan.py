# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:57:14 2021

@author: LocalAdmin
"""
import pickle as pkl
import matplotlib.pyplot as plt

from DIM.miscellaneous.PreProcessing import arrange_data_for_ident

# Load Versuchsplan to find cycles that should be considered for modelling
data = pkl.load(open('./data/Versuchsplan/Versuchsplan.pkl','rb'))


cycles = []

for i in range(1,11):
    cycles.append(pkl.load(open('./data/Versuchsplan/cycle'+str(i)+'.pkl','rb')))

_,q,_ = arrange_data_for_ident(cycles,
                               ['Durchmesser_innen'],[],[],[],'quality')

T_wkz = [cycle['T_wkz_ist'] for cycle in cycles]





# plt.close('all')

fig, axs = plt.subplots(1,2)
fig.set_size_inches((40/2.54,20/2.54))
# plt.subplot(2,2,1,title='cycle 7',xlabel='k');plt.plot(x_val[0])
# plt.subplot(2,2,3,title='cycle 8',xlabel='k');plt.plot(x_val[1])

plt.subplot(1,2,1,title='D innen',xlabel='cycle')
plt.plot([0,1,2,3,4,5,6,7,8,9],q,'d',markersize=12)

plt.subplot(1,2,2,title='T_wkz_ist',xlabel='k')
for i in range(0,len(cycles)):
    plt.plot(T_wkz[i],label=str(i))

plt.legend()
plt.show()



fig, axs = plt.subplots()
axs.plot(cycles[0]['Q_Vol_ist','V_Screw_ist','p_wkz_ist','T_wkz_ist','p_inj_ist'])
axs.plot(cycles[0][['T_wkz_ist','p_wkz_ist']])#,'V_Screw_ist','p_wkz_ist','T_wkz_ist','p_inj_ist'])
