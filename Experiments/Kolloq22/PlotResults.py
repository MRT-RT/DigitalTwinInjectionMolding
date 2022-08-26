import pandas as pd
import pickle as pkl
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import seaborn as sns

# %% Load data


MLP_1layer_T0 = pkl.load(open('./MLP_1layer_T0/Results_MLP_1layer_T0.pkl','rb'))
MLP_2layers_T0 = pkl.load(open('./MLP_2layers_T0/Results_MLP_2layers_T0.pkl','rb'))
MLP_1layer_p0 = pkl.load(open('./MLP_1layer_p0/Results_MLP_1layer_p0.pkl','rb'))
MLP_2layers_p0 = pkl.load(open('./MLP_2layers_p0/Results_MLP_2layers_p0.pkl','rb'))
MLP_2layers_static = pkl.load(open('./MLP_2layers_static/Results_MLP_2layers_static.pkl','rb'))

res = pd.concat([MLP_1layer_T0,MLP_2layers_T0,MLP_1layer_p0,
                 MLP_2layers_p0,MLP_2layers_static])

# %% Plot predictions

# Plot results static models
plt.close('all')
sns.set(font_scale = 0.7)
color = sns.color_palette()

fig,ax = plt.subplots(1,1)

#Warum man hier den Index verschieben muss weiß kein Mensch
# sns.lineplot(x = res_best['complexity']-1,y=res_best['BFR'],palette=color[0:5],
# hue=res_best['model'],legend=False,ax=ax)

stripplot = sns.boxplot(x = res['complexity'],y=res['BFR'],hue=res['model'],
                          dodge=True,ax=ax,palette=color[0:5])

legend = stripplot.get_legend()
legend.set_title(None)

legend_texts = ['$\mathrm{MLP}^{n}_{\mathrm{s}}$ 1 layer' + ' only ' + '$T_{0}$',
                '$\mathrm{MLP}^{n}_{\mathrm{s}}$ 2 layers' + ' only ' + '$T_{0}$',
                '$\mathrm{MLP}^{n}_{\mathrm{s}}$ 1 layer' + ' all ' + '$p_{0}$',
                '$\mathrm{MLP}^{n}_{\mathrm{s}}$ 2 layers' + ' all ' + '$p_{0}$',
                '$\mathrm{MLP}^{n}_{\mathrm{s}}$ 2 layers' + ' static' ]

for handle,new_text in zip(legend.texts,legend_texts):
    handle.set_text(new_text)

legend.set_bbox_to_anchor((1.05, 1))

ax.set_xlabel('$n$')
ax.set_ylabel('$\mathrm{BFR}$')

fig.set_size_inches((16/2.54,12/2.54))
# ax.set_xlim([0.8,10.2])
ax.set_ylim([0.7,1])
# ax.set_xticks(range(1,11))

plt.tight_layout()
plt.savefig('Results_Deckel_Di_MLP_zoom.png', bbox_inches='tight',dpi=600)

# %% Plot charge 131 144 (same setpoints)
