# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:19:16 2019
@author: Felix Siebenhühner
notes: python 3.7
"""

#%% IMPORT PACKAGES AND SET COLORMAPS 

source_directory    = 'code_location\\'
project_dir         = 'main_directory\\'
data_dir            = project_dir + 'Data\\'
settings_dir        = project_dir + 'Settings\\'

import sys
sys.path.append(source_directory)

import numpy as np
import rs_gen_functions as rsgfun
import parc_functions as parcfun
import surface_plotting as surf
import plot_functions as plots
import matplotlib as mpl
import bootstrap as bst
import os
import statsmodels.sandbox.stats.multicomp as multicomp
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from scipy import stats as st

plots.set_mpl_defs()

colordata3 = [[0.294, 0.0, 0.569],[0.00, 0.648, 0.569],[0.863, 0.392, 0.0]]
my_cmap3   = plots.make_cmap(colordata3)  
 
# set matplotlib parameters    
mpl.rcParams['pdf.fonttype'] = 42           
mpl.rcParams.update({'font.size': 8})
mpl.rcParams.update({'axes.titlesize': 8})
mpl.rcParams.update({'axes.labelsize': 8})
mpl.rcParams.update({'legend.fontsize': 6})
mpl.rcParams.update({'xtick.labelsize': 7})
mpl.rcParams.update({'ytick.labelsize': 7})

plots.set_mpl_defs()


#%%    ############        FREQUENCIES       ############

frequencies   = np.array([3, 3.28, 3.68, 4.02, 4.44, 5.05, 5.81, 6.56, 7.4, 8.05, 8.87, 
                          10.09, 11.55, 13.06, 14.89, 16, 17.63, 20.1, 23.1, 25.95, 30, 
                          35.43, 40, 45.0, 51.43, 60])
f_list        = [[f] for f in frequencies]
N_freq        = len(frequencies)

f_bands    = [ range(8), range(8,14), range(14,20), range(20,26)]
f_band_str = ['theta','alpha','beta','gamma1']
N_bands    = len(f_bands)

freq_strings  = ['{:.2f}'.format(f) for f in frequencies]
freq_strings0 = [str(int(round(f))) for f in frequencies]
freq_strings1 = ['{:.1f}'.format(f) for f in frequencies]

#%%    ############       LOAD SUBJECT INFO      ############


# load comt and bdnf info
comt_file    = settings_dir + 'comt.txt'
comt         = np.genfromtxt(comt_file,delimiter='\t')
comt_idx     = [list(np.where(comt==i)[0]) for i in [0,1,2]]
comt_groups  = ['Val/Val', 'Val/Met', 'Met/Met']
comt_groups2 = ['Val-Val', 'Val-Met', 'Met-Met']

bdnf_file    = settings_dir + 'bdnf.txt'
bdnf         = np.genfromtxt(bdnf_file,delimiter='\t')
bdnf_idx     = [list(np.where(bdnf==i)[0]) for i in [0,1]]
bdnf_groups  = ['Val','Met'] 





#%%      ############       LOAD  DATA       ###################


parcind   = 0         # parcellation index
dind      = 0         # data index

parcs1    = ['parc2009','Yeo7']
parcs2    = ['parc2009','parc2011yeo7']
N_parcs   = [148,16]

parc      = parcs2[parcind]
N_parc    = N_parcs[parcind]

datatypes = ['Amplitude','DFA','wPLI']
datatype  = datatypes[dind]

subdirectory  = data_dir + 'N82_rest_' + datatype +  '\\'
ylabel        = datatype

plot_folder = project_dir + 'Plots\\'


if dind==2:
    data1         = np.zeros([N_freq,82,N_parc,N_parc],'single')
    data_flat     = np.zeros([N_freq,82,N_parc*N_parc],'single')
    if parc == 'parc2009':
        DEM_file  = settings_dir + 'DEM_matrix ' + parc + '.csv' 
        DEM_used  = np.reshape(np.genfromtxt(DEM_file,delimiter=';'),148*148)
    else:
        DEM_used  = np.ones(N_parc*N_parc)
    
else:
    data1         = np.zeros([N_freq,82,N_parc],'single')
    data_flat     = np.zeros([N_freq,82,N_parc],'single')
    DEM_used      = np.ones(N_parc)
    
for f,ff in enumerate(frequencies):
     f_str         = '{:.2f}'.format(ff)
     if dind==0:
         filename      = subdirectory + 'rest hf=' + f_str + ' T0 ' + parc +  '_im.csv'
     else:
         filename      = subdirectory + 'rest hf=' + f_str + ' T0 ' + parc +  '.csv'
         
     data_flat[f]      = DEM_used * np.genfromtxt(filename, delimiter=';')
     
     if dind==2:
         data1[f]      = np.reshape(data_flat[f],(82,N_parc,N_parc))
     else:
         data1[f]      = np.reshape(data_flat[f],(82,N_parc))
         
     print(ff)


parc2009 = parcfun.Parc('parc2009')




#%% ########################################################################### 

'''    ##############       GROUP MEANS AND TESTS      ################     '''


#%%
   
'''     ##############       KRUSKAL WALLIS TEST      ################      '''


# if dind==2:
#     dataX = np.log10(data1)
# else:

dataX = data1

h_krusk_comt = np.zeros([N_freq])
p_krusk_comt = np.zeros([N_freq])
h_krusk_bdnf = np.zeros([N_freq])
p_krusk_bdnf = np.zeros([N_freq])        

for f in range(N_freq):
    
        a2   = [[np.nanmean(dataX[f][i]) for i in comt_idx[j]] for j in range(3)]
        h,p = st.kruskal(a2[0],a2[1],a2[2])
        h_krusk_comt[f] = h
        p_krusk_comt[f] = p
        
        a3   = [[np.mean(dataX[f][i]) for i in bdnf_idx[j]] for j in range(2)]
        h,p = st.kruskal(a3[0],a3[1])
        h_krusk_bdnf[f] = h
        p_krusk_bdnf[f] = p



mc_meth = 'fdr_bh'

sig_krusk_comt   = (p_krusk_comt <  0.05) * 1.
sig_krusk_comt[np.where(sig_krusk_comt < 1)] = np.nan
sig_krusk_comt_c = 1.*multicomp.multipletests(sig_krusk_comt,method=mc_meth)[0]
sig_krusk_comt_c[np.where(sig_krusk_comt_c < 1)] = np.nan

sig_krusk_bdnf   = (p_krusk_bdnf <  0.05) * 1.
sig_krusk_bdnf[np.where(sig_krusk_bdnf < 1)] = np.nan
sig_krusk_bdnf_c = 1.*multicomp.multipletests(sig_krusk_bdnf,method=mc_meth)[0]
sig_krusk_bdnf_c[np.where(sig_krusk_bdnf_c < 1)] = np.nan


#%% 

'''    ################           MAKE PLOTS         #################      '''



#%%  get averages, median over groups

p1 = 25
p2 = 75

av_comt       = [[]for ff in frequencies]
av_bdnf       = [[]for ff in frequencies]
av_str_comt   = [[]for ff in frequencies]
av_str_bdnf   = [[]for ff in frequencies]
sd_str_comt   = [[]for ff in frequencies]
sd_str_bdnf   = [[]for ff in frequencies]
p1_bdnf       = [[]for ff in frequencies]
p2_bdnf       = [[]for ff in frequencies]
p1_comt       = [[]for ff in frequencies]
p2_comt       = [[]for ff in frequencies]
med_str_comt  = [[]for ff in frequencies]
med_str_bdnf  = [[]for ff in frequencies]
comt_mean_ps  = [[]for ff in frequencies]
bdnf_mean_ps  = [[]for ff in frequencies]
comt_stats    = np.zeros([N_freq,3,3])
bdnf_stats    = np.zeros([N_freq,2,3])

for f,ff in enumerate(frequencies):
    av_comt[f]      = [ np.mean      (data1[f,comt_idx[i]],axis=0) for i in [0,1,2]]
    av_bdnf[f]      = [ np.mean      (data1[f,bdnf_idx[i]],axis=0) for i in [0,1]] 
    p1_comt[f]      = [ np.percentile(data1[f,comt_idx[i]],p1)     for i in [0,1,2]]
    p2_comt[f]      = [ np.percentile(data1[f,comt_idx[i]],p2)     for i in [0,1,2]]
    p1_bdnf[f]      = [ np.percentile(data1[f,bdnf_idx[i]],p1)     for i in [0,1]]
    p2_bdnf[f]      = [ np.percentile(data1[f,bdnf_idx[i]],p2)     for i in [0,1]]    
    av_str_comt[f]  = [ np.mean      (data1[f,comt_idx[i]])        for i in [0,1,2]]
    av_str_bdnf[f]  = [ np.mean      (data1[f,bdnf_idx[i]])        for i in [0,1]]
    sd_str_comt[f]  = [ np.std       (data1[f,comt_idx[i]])        for i in [0,1,2]]
    sd_str_bdnf[f]  = [ np.std       (data1[f,bdnf_idx[i]])        for i in [0,1]]
    med_str_comt[f] = [ np.median    (data1[f,comt_idx[i]])        for i in [0,1,2]]
    med_str_bdnf[f] = [ np.median    (data1[f,bdnf_idx[i]])        for i in [0,1]]
    
    if dind == 2:
        comt_mean_ps[f] = [ np.mean      (data1[f,comt_idx[i]],(1,2))  for i in [0,1,2]]
        bdnf_mean_ps[f] = [ np.mean      (data1[f,bdnf_idx[i]],(1,2))  for i in [0,1]]
    else:
        comt_mean_ps[f] = [ np.mean      (data1[f,comt_idx[i]],(1))  for i in [0,1,2]]
        bdnf_mean_ps[f] = [ np.mean      (data1[f,bdnf_idx[i]],(1))  for i in [0,1]]
        
    comt_stats[f]       = [bst.CI_from_bootstrap(comt_mean_ps[f][i])[:3] for i in [0,1,2]]  
    bdnf_stats[f]       = [bst.CI_from_bootstrap(bdnf_mean_ps[f][i])[:3] for i in [0,1]]  
 


#%% generate plot data and make  plots

plot_data_comt = np.zeros([3,3,N_freq])
plot_data_bdnf = np.zeros([2,3,N_freq])

export = False
filetype = 'pdf'
plot_data = 'CI'

if plot_data == 'SD':
    for f,ff in enumerate(frequencies):
        for i in range(3):
            plot_data_comt[i,:,f] = [av_str_comt[f][i], av_str_comt[f][i]-sd_str_comt[f][i],  \
                           av_str_comt[f][i] + sd_str_comt[f][i] ]
        for i in range(2):
            plot_data_bdnf[i,:,f] = [av_str_bdnf[f][i], av_str_bdnf[f][i]-sd_str_bdnf[f][i],  \
                           av_str_bdnf[f][i] + sd_str_bdnf[f][i] ] 
elif plot_data == 'pct':
    for f,ff in enumerate(frequencies):
        for i in range(3):
            plot_data_comt[i,:,f] = [av_str_comt[f][i],p1_comt[f][i],p2_comt[f][i]]
        for i in range(2):
            plot_data_bdnf[i,:,f] = [av_str_bdnf[f][i],p1_bdnf[f][i],p2_bdnf[f][i]]  
            
elif plot_data == 'CI':
    for f,ff in enumerate(frequencies):
        for i in range(3):
            plot_data_comt[i,:,f] = comt_stats[f][i]
        for i in range(2):
            plot_data_bdnf[i,:,f] = bdnf_stats[f][i]
        

ylims       = [ [10, 26] ,      [0.55, 0.75], [0,0.155]    ]
ylim        = ylims[dind]

if export:
    outfile1    = plot_folder + 'Mean values ' + datatype + '\\COMT.' + filetype
    outfile2    = plot_folder + 'Mean values ' + datatype + '\\BDNF.' + filetype
else:
    outfile1 = outfile2 = 'none'


plots.semi_log_plot([6,2.5],plot_data_comt[:3,:3,:],frequencies,[3,60],
                    ylabel,legend=None,legend_pos=None,show=True,
                    cmap=my_cmap3,CI=0.25, xticks=[3,5,10,20,30,60,120],
                    ylim=ylim,outfile=outfile1,bgcolor=(1,1,1),ncols=1,
                    sig_id=sig_krusk_comt,sig_fac=1.2
                    )


plots.semi_log_plot([6,2.5],plot_data_bdnf[:3,:3,:],frequencies,[3,60],
                    ylabel,legend=None,#['Val/Val', 'Val/Met & Met/Met'],
                    legend_pos=None,show=True,
                    cmap=my_cmap3,CI=0.25,xticks=[3,5,10,20,30,60,120],
                    ylim=ylim,outfile=outfile2,bgcolor=(1,1,1),ncols=1,
                    sig_id=sig_krusk_bdnf,sig_style='-',sig_fac=1.02
                    )





















#%%


'''    ###############       SYNCH NETWORK ANALYSIS     ################    '''


#%%

'''  ###############   PLOT synch data in yeo 7 networks    ##############  '''


#%% load networks

networks_file         = settings_dir + 'networks ' + parc + '.csv'
networks              = np.genfromtxt(networks_file,delimiter=';')

if parc == 'parc2009':
    for i in [ 28,  29,  30,  31, 106, 144, 145]:
        networks[i] = -1

network_names         = ['FP','DM','SM','Lim','VA','DA','Vis']
network_idx           = [np.array(np.where(networks==i)[0])   for i in range(7)]
network_masks_within  = [np.outer(networks==i,networks==i)    for i in range(7)]
network_masks_all     = [[np.outer(networks==i,networks==j)   for i in range(7)] for j in range(7)]
network_mask_sums     = [[np.sum(np.reshape(DEM_used,[N_parc,N_parc])*network_masks_all[i][j]) for i in range(7)] for j in range(7)]

network_l = [a[0:1] for a in network_idx]  
network_r = [a[1:2] for a in network_idx]


#%% morph to yeo7 and collapse in frequency bands



data1a     = np.zeros([N_freq,82,7,7],'single')


# select hemi    
hemi = 'cross'                   # can be "all", "left", "right", "cross" 
        
if hemi == "cross":                    
    nix1 = network_l
    nix2 = network_r
elif hemi == 'left':
    nix1 = network_l
    nix2 = network_l                
elif hemi == 'right':
    nix1 = network_r
    nix2 = network_r
else:                        
    nix1 = nix2 = network_idx
                    
for f in range(N_freq):
    for s in range(82):
        for n1 in range(7):
            for n2 in range(7):
                dummy =  data1[f,s,nix1[n1],:]  
                data1a[f,s,n1,n2] =  np.mean(dummy[:,nix2[n2]])


# collapse to freq bands        
 
mean_comt     = np.array([[ np.mean (data1a[f,comt_idx[i]],axis=0) for i in [0,1,2]] 
                          for f in range(N_freq)])
mean_bdnf     = np.array([[ np.mean (data1a[f,bdnf_idx[i]],axis=0) for i in [0,1]  ] 
                          for f in range(N_freq)])
                 
mean_comt_pf  = np.array([ np.mean(mean_comt[fb],0) for fb in f_bands])
mean_bdnf_pf  = np.array([ np.mean(mean_bdnf[fb],0) for fb in f_bands])


#%%

save_datatype = '.eps'

#%% plot COMT values in 7 networks


vmaxA = [0.075,0.105,0.045,0.06, 0.06]
fig   = rsgfun.plot_comt_values(N_bands,mean_comt_pf,f_band_str,comt_groups,network_names,vmaxA)
 

fig.savefig(plot_folder + 'synch\\COMT_values_' + hemi + save_datatype, 
       )    
    
#%% plot BDNF values  in 7 networks  

vmaxA = [0.09, 0.105, 0.048, 0.054, 0.054]
fig   = rsgfun.plot_bdnf_values(N_bands,mean_bdnf_pf,f_band_str,bdnf_groups,network_names,vmaxA)

fig.savefig(plot_folder + 'synch\\BDNF_values_' + hemi + save_datatype)    
 
    
#%% plot COMT differences  

vmaxA = [0.010, 0.012, 0.004, 0.010]
fig   = rsgfun.plot_comt_diff(N_bands,mean_comt_pf,f_band_str,comt_groups,network_names,vmaxA)

fig.savefig(plot_folder + 'synch\\COMT_diffs_' + hemi + save_datatype)    


#%% plot BDNF differences 

vmaxA = [0.018, 0.03, 0.006, 0.006]
fig   = rsgfun.plot_bdnf_diff(N_bands,mean_bdnf_pf,f_band_str,bdnf_groups,network_names,vmaxA)

fig.savefig(plot_folder + 'synch\\BDNF_diffs_' + hemi + save_datatype)    


#%% kruskal-wallis for COMT

fig = rsgfun.synch_test_comt(data1a,N_bands,f_bands,comt_idx,network_names,f_band_str)
 



#%% kruskal-wallis for BDNF

fig = rsgfun.synch_test_bdnf(data1a,N_bands,f_bands,bdnf_idx,network_names,f_band_str)

fig.savefig(plot_folder + 'synch\\BDNF_diff_sign_' + hemi + save_datatype)    






#%%



'''  ###########            Amp & DFA analysis by subsystem and f-band          ##############  '''


#%% 

if datatype == 'Amplitude':
    ymin    = 0
    ymax   = [25,35,35,25]
else:
    ymin = 0.5
    ymax = [0.8,0.9,0.8,0.8]
    

# COMT
data1_gr = np.stack([np.nanmean(data1[:,inds],1) for inds in comt_idx])
outdir = project_dir + 'Plots\\__plot_data\\boxplots ' + datatype + '\\COMT\\'
colors = [[0.294, 0.0, 0.569],[0.00, 0.648, 0.569],[0.863, 0.392, 0.0]]

fig,axes = plt.subplots(4,7,figsize=[25,18])
for n in range(7):    
    for fb in range(4):
        ax = axes[fb,n]
        data1_gr_fb = np.array([np.nanmean(data1_gr[:,f_bands[fb]],1)])[0]
        bplot = ax.boxplot(np.transpose(data1_gr_fb[:,parc2009.netw_indices[n]]), patch_artist=True)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        if fb<3:
            ax.set_xticks([])
        else:
            ax.set_xticklabels(comt_groups)
        if n>0:
            ax.set_yticks([])
        if fb == 0:
            ax.set_title(parc2009.netw_names[n],fontsize=17)
        ax.set_ylim([ymin,ymax[fb]])
        
        np.savetxt(outdir + f_band_str[fb] + ' ' + parc2009.netw_names[n] + '.csv',data1_gr_fb ,delimiter=';')


fig.savefig(project_dir + 'Plots\\Boxplots Amplitude, DFA\\Boxplots ' + datatype + ' COMT, mean over subjects.eps')

#BDNF
data1_gr = np.stack([np.nanmean(data1[:,inds],1) for inds in bdnf_idx])
outdir = project_dir + 'Plots\\__plot_data\\boxplots ' + datatype + '\\BDNF\\'
colors = [[0.294, 0.0, 0.569],[0.863, 0.392, 0.0]]

fig,axes = plt.subplots(4,7,figsize=[25,18])
for n in range(7):    
    for fb in range(4):
        ax = axes[fb,n]
        data1_gr_fb = np.array([np.nanmean(data1_gr[:,f_bands[fb]],1)])[0]
        bplot = ax.boxplot(np.transpose(data1_gr_fb[:,parc2009.netw_indices[n]]), patch_artist=True)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        if fb<3:
            ax.set_xticks([])
        else:
            ax.set_xticklabels(bdnf_groups)
        if n>0:
            ax.set_yticks([])
        if fb == 0:
            ax.set_title(parc2009.netw_names[n],fontsize=17)
        ax.set_ylim([ymin,ymax[fb]])
        
        np.savetxt(outdir + f_band_str[fb] + ' ' + parc2009.netw_names[n] + '.csv',data1_gr_fb ,delimiter=';')


fig.savefig(project_dir + 'Plots\\Boxplots Amplitude, DFA\\Boxplots ' + datatype + ' BDNF, mean over subjects.eps')






#%% re-arrange data 

parcels = np.genfromtxt(settings_dir + 'parcels.csv',dtype='str',delimiter=';')[:,0]

parcel_conv = np.genfromtxt(settings_dir + 'parc2009_py_vs_LV.csv',dtype='str',delimiter=';')
new_inds = [np.where(p==parcel_conv[1:,1])[0][0] for p in parcel_conv[1:,0]]


parc_name = 'parc2009'
parc_dir  = 'L:\\nttk_palva\\Utilities\\parcellations\\parc_info\\'
parc2009  = parcfun.Parc(parc_name,dir1=parc_dir,suffix='')

f_bands = [ range(8), range(8,14), range(14,20), range(20,26)]
f_band_str = ['01 theta','02 alpha','03 beta','04 gamma']


data2 = np.stack([[dd[new_inds] for dd in d] for d in data1])

plot_folder = project_dir + 'Plots\\'




#%% define groups and get data in frequency bands

groups = [comt_idx[0],comt_idx[1],comt_idx[2],bdnf_idx[0],bdnf_idx[1]]
group_names = ['Val-Val', 'Val-Met', 'Met-Met', 'Val', 'Met']
pair_names  = ['Val-Met - Met-Met','Val-Met - Val-Val','Val-Val - Met-Met', 'Val - Met']
pair_names2 = ['Met-Met - Val-Met','Val-Val - Val-Met','Met-Met - Val-Val', 'Met - Val']

group_pairs = [
    [comt_idx[1],comt_idx[2]],
    [comt_idx[1],comt_idx[0]],    
    [comt_idx[0],comt_idx[2]],
    [bdnf_idx[0],bdnf_idx[1]]]

N_pairs  = len(group_pairs)
N_groups = len(groups)

f_band_data = np.stack([np.mean(data2[f_inds],0) for f_inds in f_bands ])

plot_data = np.stack([np.mean(f_band_data[:,gr_inds],1) for gr_inds in groups])


if datatype == 'DFA':
    zmax = [0.64,0.71,0.71,0.64]
    zmin = 0.5
else:
    zmax=[20,30,25,25]
    zmin=10
    


    
#%% plot means per band and group on brain surfaces
    
maxA = np.zeros([5,4])    
minA = np.zeros([5,4]) 
    
for gr in range(5):
    for f in range(4):    
        minA[gr,f] = np.min(plot_data[gr][f])
        maxA[gr,f] = np.max(plot_data[gr][f])        
        
        if gr<3:
            gr_type = 'COMT'
        else:
            gr_type = 'BDNF'
            
        title    = f_band_str[f] + ' ' + group_names[gr]
        folder1  = plot_folder + 'Brain plots ' + datatype + '\\'
        filename = folder1 + gr_type + '\\' + title  
        
        surf.plot_4_view(plot_data[gr][f],parc2009.names,'parc2009',alpha=1,
                     zmax=zmax[f],zmin=zmin,style='linear', title=title,thresh=0.0001,
                     filename = filename,transparent=0,cmap='gnuplot',show=True) 





#%% compute group means and differences

test_type = 'MWU'          

diffs = np.zeros([N_freq,N_pairs,N_parc])
stats = np.zeros([N_freq,N_pairs,N_parc])
pvals = np.zeros([N_freq,N_pairs,N_parc])
K_pos = np.zeros([N_pairs,N_freq])
K_neg = np.zeros([N_pairs,N_freq])
means = np.zeros([N_freq,N_groups,N_parc])

for f in range(N_freq):
    data_f = data1[f]
    for g in range(N_groups):
        means[f,g] = np.mean(data_f[groups[g]],0)
    
    for g in range(N_pairs):
        groups12 = group_pairs[g]
        gr1 = data_f[groups12[0]]
        gr2 = data_f[groups12[1]]
        for p in range(N_parc):            
            diffs[f,g,p] = np.mean(gr1[:,p]) - np.mean(gr2[:,p])
            
            if test_type == 'Welch':
                stats[f,g,p], pvals[f,g,p] = ttest_ind(gr1[:,p],gr2[:,p])
            elif test_type == 'Kruskal':                  
                stats[f,g,p], pvals[f,g,p] = kruskal(gr1[:,p],gr2[:,p])
            elif test_type == 'MWU':
                stats[f,g,p], pvals[f,g,p] = mannwhitneyu(gr1[:,p],gr2[:,p])

                        
alpha = 0.05

K_pos = 100*np.sum((diffs>0)*(pvals<alpha),2)/N_parc
K_neg = 100*np.sum((diffs<0)*(pvals<alpha),2)/N_parc

K_all = K_pos + K_neg   

    

#%% plot P for COMT

if datatype=='DFA':
    ylim = [-30, 90]
    yticks = np.arange(-20,81,20)    
else:
    ylim = [-27, 27]
    yticks = np.arange(-20,26,10) 
    
save_filetype = 'none'# '.eps'             # '.eps' or '.png' or 'none'



Q= 0.03

if datatype=='DFA':
    ylim = [-30, 90]
    yticks = np.arange(-20,81,20)    
else:
    ylim = [-27, 27]
    yticks = np.arange(-20,26,10)    

xticks = [3, 5, 10, 20, 30, 60]
    
Q_lim = 50*np.repeat((alpha+Q),N_freq)

plot_data = np.transpose(np.concatenate([K_pos[:,:3],-K_neg[:,:3],
                                         np.expand_dims(Q_lim,1),np.expand_dims(-Q_lim,1)],1))
cmap = plots.make_cmap([(1,.5,0),(0,0,1),(1,0,1),(1,.5,0),(0,0,1),(1,0,1),(.4,.4,.4),(.4,.4,.4)])
legend=['Val/Met - Met/Met','Val/Met - Val/Val','Val/Val - Met/Met']

if save_filetype == 'none':
    outfile = 'none'
else:
    outfile = plot_folder + 'P+- plots ' + datatype + '\\COMT' + save_filetype

plots.semi_log_plot([7,3],plot_data,frequencies,xlim=[3,60],ylabel='P [%]',
                    legend=legend,outfile=outfile,xticks=xticks,
                    ylim=ylim,yticks=yticks,title='COMT',cmap=cmap,ncols=1)    

datafile_name = plot_folder + '__plot_data\\P_plots\\' + datatype + ' COMT.csv'
np.savetxt(datafile_name,np.around(plot_data[:6],4),delimiter=';',fmt='%.4f')



#%% plot P for BDNF

plot_data = np.transpose(np.concatenate([K_pos[:,3:],-K_neg[:,3:],np.expand_dims(Q_lim,1),np.expand_dims(-Q_lim,1)],1))
cmap=plots.make_cmap([(0,0,1),(0,0,1),(.4,.4,.4),(.4,.4,.4)])


if save_filetype == 'none':
    outfile = 'none'
else:
    outfile = plot_folder + 'P+- plots ' + datatype + '\\BDNF' + save_filetype

plots.semi_log_plot([7,3],plot_data,frequencies,xlim=[3,60],ylabel='P [%]',
                    legend=['Val - Met'],ylim=ylim,yticks=yticks,xticks=xticks,
                    title='BDNF',cmap=cmap,ncols=1,outfile=outfile)    

datafile_name = plot_folder + '__plot_data\\P_plots\\' + datatype + ' BDNF.csv'
np.savetxt(datafile_name,np.around(plot_data[:2],4),delimiter=';',fmt='%.4f')


#%% make plots in selected freq. ranges

save_datatype = '.png'          # '.eps' or '.png' or 'none'

thresh_type = 'percentile'         



if datatype=='DFA':
    zmax=0.07
else:
    zmax=5

zmax = None

thresh_if  = False
thresh_pct = True
 
if datatype == 'Amplitude':
    combs = [
        [0,'pos',range( 0, 8)],[0,'pos',range( 8,17)],[0,'neg',range(17,26)],
        [1,'pos',range( 0,11)],                       [1,'neg',range(18,26)],
                                                      [2,'neg',range(17,26)],
        [3,'pos',range( 0, 8)],[3,'pos',range( 8,16)],[3,'neg',range(18,26)]
        ]
else:
    combs = [
        [0,'pos',range( 0,12)], [0,'pos',range(17,26)],
        [1,'pos',range(16,26)],
        [2,'pos',range( 0,10)],
        [3,'pos',range( 0, 8)], [3,'pos',range( 8,16)],
        ]
    
        

for c in range(len(combs)):
    comb   = combs[c]
    gr     = comb[0]
    tail   = comb[1]
    f_inds = comb[2]
    freqs  = frequencies[f_inds]
    
    if gr<3:
        gr_type = 'COMT'
    else:
        gr_type = 'BDNF'
    pd0 = np.mean(diffs[f_inds,gr],0)

    if tail == 'pos':
        title = pair_names[gr] + ', ' + str(int(np.floor(min(freqs)))) + ' - ' + str(int(np.around(max(freqs)))) + ' Hz'
        cmap  = 'Reds'
    else:
        pd0   = pd0 * -1       
        title = pair_names[gr] + ', ' + str(int(np.floor(min(freqs)))) + ' - ' + str(int(np.ceil(max(freqs)))) + ' Hz'
        cmap  = 'Blues'
    
    if thresh_type == 'percentile':
            pd1 = pd0 * (pd0>np.percentile(pd0,80))  
    folder1  = plot_folder + 'Brain plots ' + datatype + '\\'
    filename = folder1 + gr_type + ' diff\\' + title + save_datatype
    zmax     = surf.find_7max(np.max(pd1),'down')
    
    inds = np.argsort(pd1)[-15:][::-1]
    print(title)
    print(*([parc2009.abbr[i] + ' : ' +  str(np.round(pd1[i],2)) for i in inds]),sep='\n')
    print('\n')
   
    
    
    surf.plot_4_view(pd0,parc2009.names,'parc2009',alpha=1,
                 zmax=zmax,zmin=0,style='linear', title=title,thresh=0.0001,
                 filename=None,transparent=0,show=True) 



#%% export data for use in combined plots

dir1 = plot_folder + '__plot_data\\_per_band\\' + datatype + ' diffs_sig_any\\'
os.makedirs(dir1,exist_ok=True)

if datatype == 'Amplitude':
    combs = [
        [0,'pos',range( 0, 8)],[0,'pos',range( 8,17)],[0,'neg',range(17,26)],
        [1,'pos',range( 0, 8)],                       [1,'neg',range(17,26)],
                                                      [2,'neg',range(17,26)],
        [3,'pos',range( 0, 8)],[3,'pos',range( 8,17)],[3,'neg',range(17,26)]
        ]
else:
    combs = [
        [0,'pos',range( 0, 8)],                       [0,'pos',range(17,26)],
                                                      [1,'pos',range(17,26)],
        [2,'pos',range( 0, 8)],
        [3,'pos',range( 0, 8)], [3,'pos',range( 8,17)],
        ]


for c in range(len(combs)):
    comb   = combs[c]
    gr     = comb[0]
    tail   = comb[1]
    f_inds = comb[2]
    freqs  = frequencies[f_inds]
    
    if gr<3:
        gr_type = 'COMT'
    else:
        gr_type = 'BDNF'
        
    pd0  = np.mean(diffs[f_inds,gr],0)
    sig0 = np.sum((pvals[f_inds,gr]<0.05),0)>0
    
    out_data = pd0 * sig0
    
    title = pair_names[gr] + ', ' + str(int(np.floor(min(freqs)))) + ' - ' + str(int(np.ceil(max(freqs)))) + ' Hz'
    file1 = dir1 + gr_type + ' ' + title + '.csv'

    np.savetxt(file1,out_data)
















