
"""
Created on Wed May  4 11:34:38 2022

@author: LocalAdmin
"""

import pandas as pd
import matplotlib.pyplot as plt
from time import sleep

plt.close('all')
plt.style.use('ggplot')



csv_file_path = 'C:/Users/LocalAdmin/Downloads/Messdaten_Verschlusskappe.csv'
qual_label = 'Durchmesser_innen'

window_with = 100 

df_plot = pd.DataFrame(columns=[qual_label])

# Initialize plots
fig = plt.figure()

colors = [plt.get_cmap('tab10')(i) for i in range(0,8)]

D_axis = plt.axes([0.05, 0.05, 1, 0.4])
D_mean_axis = plt.axes([0.05, 0.5, 1, 0.4])

D_line, = D_axis.plot([],[],'-o',alpha=0.8)

D_mean, = D_mean_axis.plot([],[],'-o',alpha=0.8)

plt.show()

t_mean = []
mean = []

for k in range(3,100):
    
      
    # read csv
    df_csv = pd.read_csv(csv_file_path,sep=';',index_col=0, 
                     encoding_errors='ignore')

    
    # k = df_csv['laufenden Zhler'].max()
    current_qual_meas = df_csv[df_csv['laufenden Zhler']==k][qual_label]
    current_qual_meas = float(current_qual_meas.values[0].replace(',','.'))
    
    
    if  k != df_plot.index.max():
        df_plot.loc[k] = current_qual_meas

        # move window 
        t_min = max(0,k-window_with)
        t_max = max(0,k-window_with)+window_with
        
        t = df_plot.index[t_min:t_max]
        
        D_line.set_data(t,df_plot[qual_label].loc[t])
        
        D_axis.set_xlim([t_min,t_max])
        D_axis.set_ylim([df_plot[qual_label].min()-0.1,df_plot[qual_label].max()+0.1])
    
    
        if k % 10 == 0:
            t_mean.append(k)
            mean.append(df_plot[qual_label].loc[k-10:k].mean())
            
            D_mean.set_data(t_mean,mean)
            
            D_mean_axis.set_xlim([t_min,t_max])
            D_mean_axis.set_ylim([df_plot[qual_label].min()-0.1,df_plot[qual_label].max()+0.1])
    
        plt.pause(0.5)
        sleep(4)