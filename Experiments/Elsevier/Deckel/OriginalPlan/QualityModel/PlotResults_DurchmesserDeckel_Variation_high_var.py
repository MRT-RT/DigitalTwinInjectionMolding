import pandas as pd
import pickle as pkl
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import seaborn as sns

# Result of models copy and pasted from Eval_..._.py

poly_setpoints = pkl.load(open('./setpoint_models/Durchmesser_innen/setpoints/Poly_set_Durchmesser_high_var.pkl','rb'))
poly_setpoints_x0 =    pkl.load(open('./setpoint_models/Durchmesser_innen/setpoints_initial_state/Poly_set_x0_Durchmesser_high_var.pkl','rb'))
# MLP_setpoints = pkl.load(open('./setpoint_models/Durchmesser_innen/setpoints/MLP_set_Durchmesser_high_var.pkl','rb'))
# MLP_setpoints_x0 = pkl.load(open('./setpoint_models/Durchmesser_innen/setpoints_initial_state/MLP_set_x0_Durchmesser_high_var.pkl','rb'))
MLP_setpoints = pkl.load(open('./setpoint_models/Durchmesser_innen/MLP_two_layers/MLP_2layer_set_Durchmesser_high_var.pkl','rb'))
MLP_setpoints_x0 = pkl.load(open('./setpoint_models/Durchmesser_innen/MLP_two_layers_initial_state/MLP_2layer_set_x0_Durchmesser_high_var.pkl','rb'))
GRU = pkl.load(open('./GRU/Durchmesser_innen/GRU_3sub_Durchmesser_innen_high_var.pkl','rb'))

res = pd.concat([poly_setpoints,poly_setpoints_x0,MLP_setpoints,
                 MLP_setpoints_x0,GRU])

poly_setpoints_best = poly_setpoints
poly_setpoints_x0_best = poly_setpoints_x0
MLP_setpoints_best = MLP_setpoints.sort_values('BFR',ascending=False).groupby('complexity').head(1)
MLP_setpoints_x0_best = MLP_setpoints_x0.sort_values('BFR',ascending=False).groupby('complexity').head(1)
GRU_best = GRU.sort_values('BFR',ascending=False).groupby('complexity').head(1)

res_best = pd.concat([poly_setpoints_best,poly_setpoints_x0_best,
                      MLP_setpoints_best,MLP_setpoints_x0_best,GRU_best])

# res_best = poly_setpoints_best
res_best.index = range(0,len(res_best))



# Plot results static models
plt.close('all')
sns.set(font_scale = 0.7)
color = sns.color_palette()

fig,ax = plt.subplots(1,1)

#Warum man hier den Index verschieben muss weiß kein Mensch
sns.lineplot(x = res_best['complexity']-1,y=res_best['BFR'],palette=color[0:5],
hue=res_best['model'],legend=False,ax=ax)

stripplot = sns.boxplot(x = res['complexity'],y=res['BFR'],hue=res['model'],
                          dodge=True,ax=ax,palette=color[0:5])

legend = stripplot.get_legend()
legend.remove()


ax.set_xlabel('$n$')
ax.set_ylabel('$\mathrm{BFR}$')

fig.set_size_inches((8.4/2.54,6/2.54))
# ax.set_xlim([0.8,10.2])
ax.set_ylim([0.7,1])
# ax.set_xticks(range(1,11))

plt.tight_layout()
plt.savefig('Results_Deckel_Durchmesser_high_var.png', bbox_inches='tight',dpi=600)

