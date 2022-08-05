# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:56:46 2021
@author: Felix Siebenhühner
notes: python 3.7
"""


source_directory    = 'code_location\\'
project_dir         = 'main_directory\\'


source_directory    = 'K:\\palva\\Felix\\Python37\\Data analysis\\Brattico PS\\repo\\'
project_dir         = 'L:\\nttk_palva\\Projects\\Brattico\\'

data_dir            = project_dir + 'Data\\'
settings_dir        = project_dir + 'Data\\Settings\\'

import sys
sys.path.append(source_directory)

import numpy as np
import os
import pickle as pick
import matplotlib.pyplot as plt
import stats_functions as statfun
import random

amp_data   = pick.load(open(data_dir+'Data_compressed\\amp_data.dat','rb'))
dfa_data   = pick.load(open(data_dir+'Data_compressed\\dfa_data.dat','rb'))
synch_data = pick.load(open(data_dir+'Data_compressed\\synch_data.dat','rb'))




#%%    ############        FREQUENCIES       ############



frequencies   = np.array([3, 3.28, 3.68, 4.02, 4.44, 5.05, 5.81, 6.56, 7.4, 8.05, 8.87, 
                          10.09, 11.55, 13.06, 14.89, 16, 17.63, 20.1, 23.1, 25.95, 30, 
                          35.43, 40, 45.0, 51.43, 60])
f_list        = [[f] for f in frequencies]
N_freq        = len(frequencies)

freq_strings  = ['{:.2f}'.format(f) for f in frequencies]
freq_strings0 = [str(int(round(f))) for f in frequencies]
freq_strings1 = ['{:.1f}'.format(f) for f in frequencies]





#%% group means per parcel (pair)

# load comt and bdnf info

comt_file    = settings_dir + 'comt.txt'
comt         = np.genfromtxt(comt_file,delimiter='\t')
comt         = np.delete(comt,60)
comt_idx     = [list(np.where(comt==i)[0]) for i in [0,1,2]]
comt_groups  = ['Val/Val', 'Val/Met', 'Met/Met']
comt_groups2 = ['Val-Val', 'Val-Met', 'Met-Met']

bdnf_file    = settings_dir + 'bdnf.txt'
bdnf         = np.genfromtxt(bdnf_file,delimiter='\t')
bdnf         = np.delete(bdnf,60)
bdnf_idx     = [list(np.where(bdnf==i)[0]) for i in [0,1]]
bdnf_groups  = ['Val','Met'] 


gr_mean_amp_comt = np.stack([np.mean(amp_data[:,inds],1) for inds in comt_idx])
gr_mean_amp_bdnf = np.stack([np.mean(amp_data[:,inds],1) for inds in bdnf_idx])

gr_mean_dfa_comt = np.stack([np.mean(dfa_data[:,inds],1) for inds in comt_idx])
gr_mean_dfa_bdnf = np.stack([np.mean(dfa_data[:,inds],1) for inds in bdnf_idx])

gr_mean_syn_comt = np.stack([np.mean(synch_data[:,inds],1) for inds in comt_idx])
gr_mean_syn_bdnf = np.stack([np.mean(synch_data[:,inds],1) for inds in bdnf_idx])


s_mean_amp = np.mean(amp_data,(2))
s_mean_dfa = np.mean(dfa_data,(2))
s_mean_syn = np.mean(synch_data,(2,3))







#%%% correlation across subjects, with CL

corr_sel = 'Synch-DFA'
grouping = 'BDNF'
export   = False


f_bands    = [ range(0,8), range(8,14), range(14,17), range(17,20), range(20,26)]
f_band_str = ['theta','alpha','beta1','beta2','gamma'] 

N_fb = len(f_bands)

# load values and groupings
  
if corr_sel  == 'Amp-DFA':
    values2  = amp_data
    values1  = dfa_data
    name2    = 'Mean Ampl'
    name1    = 'Mean DFA'

elif corr_sel == 'Synch-DFA':
    values2  = np.nanmean(synch_data,3)
    values1  = dfa_data
    name2    = 'Mean wPLI'
    name1    = 'Mean DFA'
    
    nc = 7
    nr = 4     

if grouping == 'COMT':
    gr_ind = comt_idx
    names  = comt_groups2
    colors = [[0.294, 0.0, 0.569],[0.00, 0.648, 0.569],[0.863, 0.392, 0.0]]
    gr_pairs = [[0,1],[0,2],[1,2]]
    pair_names = ['VV - VM','VV - MM','VM - MM']

else:
    gr_ind = bdnf_idx
    names  = bdnf_groups
    colors = [[0.294, 0.0, 0.569],[0.863, 0.392, 0.0]]
    gr_pairs = [[0,1]]
    pair_names = ['Val - Met']


N_gr = len(gr_ind)


# compute original and permuted correlations
  
valA = [[values1[f,inds] for inds in gr_ind] for f in range(N_freq)]
valB = [[values2[f,inds] for inds in gr_ind] for f in range(N_freq)]
          

corrs  = np.full([N_freq,N_gr,148],np.nan)
corrsP = np.full([N_freq,N_gr,148,1000],np.nan)
method = 'Spearman'
         
for f in range(N_freq):           
    for g in range(N_gr):
        gr_size = len(gr_ind[g])
        for p in range(148):
            x1 = valA[f][g][:,p]
            y1 = valB[f][g][:,p]
            r,pv   = statfun.correlate(x1,y1,method=method)
            corrs[f,g,p] = r           
            
            
for k in range(1000): 
    for g in range(N_gr):
        gr_size = len(gr_ind[g])
        inds = np.random.randint(0,gr_size,gr_size)
        for f in range(N_freq):           
            for p in range(148):
                x1 = valA[f][g][inds,p]
                y1 = valB[f][g][inds,p]                
                r,pv   = statfun.correlate(x1,y1,method=method)
                corrsP[f,g,p,k] = r
    if k%20 == 0:
        print(k)
    
            
            
# get lower and upper percentiles
         
corr_means = np.zeros([N_gr,N_fb])
corr_CI    = np.zeros([N_gr,2,N_fb])
lowers     = np.zeros([N_gr,N_fb])
uppers     = np.zeros([N_gr,N_fb])


for f in range(N_fb):   
    corrs2  = np.array([corrs[f_bands[f],gr] for gr in range(N_gr)])
    corrs2P = np.array([corrsP[f_bands[f],gr] for gr in range(N_gr)])
    corrs2  = np.nanmean(corrs2,(1,2))
    corrs2P = np.nanmean(corrs2P,(1,2))
    corr_means[:,f] = corrs2 
    mean_perm   = np.mean(corrs2P,1)
    lowers[:,f] = np.percentile(corrs2P,2.5,1)
    uppers[:,f] = np.percentile(corrs2P,97.5,1)

lowers2 = corr_means - lowers
uppers2 = uppers - corr_means

# plot means and percentiles

plt.errorbar(np.arange(N_fb),    corr_means[0],yerr=[lowers2[0],uppers2[0]],marker='o',ls = 'none')
plt.errorbar(np.arange(N_fb)+0.2,corr_means[1],yerr=[lowers2[1],uppers2[1]],marker='o',ls = 'none')
if grouping == 'COMT':
    plt.errorbar(np.arange(N_fb)+0.4,corr_means[2],yerr=[lowers2[2],uppers2[2]],marker='o',ls = 'none')

plt.legend(names)
plt.xticks(np.arange(N_fb),f_band_str)




# permutation group comparison
corrs_X = np.full([N_freq,N_gr,148,1000],np.nan)

for k in range(1000):
    indsR = np.array(range(82))
    random.shuffle(indsR)
    valA = [[values1[f,indsR[inds]] for inds in gr_ind] for f in range(N_freq)]
    valB = [[values2[f,indsR[inds]] for inds in gr_ind] for f in range(N_freq)]
    for f in range(N_freq):           
        for g in range(N_gr):
            gr_size = len(gr_ind[g])
            for p in range(148):
                x1 = valA[f][g][:,p]
                y1 = valB[f][g][:,p]
                r,pv   = statfun.correlate(x1,y1,method=method)
                corrs_X[f,g,p,k] = r 
    if k%20 == 0:
        print(k)
    
diffs   = np.zeros([N_gr,N_fb]) 
diffs_X = np.zeros([N_gr,N_fb,1000])      
for f in range(N_fb):   
    corrs2  = np.array([corrs[f_bands[f],gr] for gr in range(N_gr)])
    corrs2X = np.array([corrs_X[f_bands[f],gr] for gr in range(N_gr)])
    corrs2  = np.nanmean(corrs2,(1,2))
    corrs2X = np.nanmean(corrs2X,(1,2)) 
    for g in range(len(gr_pairs)):
        pair = gr_pairs[g]
        i = pair[0]
        j = pair[1]
        diff    = corrs2[i] -corrs2[j]
        diff_X  = corrs2X[i]-corrs2X[j]
        print(f_band_str[f] + ' ' + pair_names[g] + ': ' + str(np.sum(diff_X>diff)/1000))
        
    
    
    
# export data
    
corr_mean_f = np.nanmean(corrs,2)


dir5 = project_dir + '\\Plots\\__plot_data\\correlation_plots_data\\' 
dir6 = dir5 + corr_sel + '_' + grouping +'\\'
os.makedirs(dir6,exist_ok=True)
    
for g in range(N_gr):
    file4 = dir6 + 'All corrs ' + names[g] + '.csv' 
    np.savetxt(file4,corrs[:,g],delimiter=';')    

for f in range(N_fb):
    file3 = dir6 + 'Mean corrs ' + f_band_str[f] + '.csv' 
    np.savetxt(file3,corr_mean_f[f_bands[f]],delimiter=';')
        
    
    
file5 = dir6 + 'Grand mean per f-band.csv' 
file6 = dir6 + 'CL lower per f-band.csv' 
file7 = dir6 + 'CL upper per f-band.csv'  
    
np.savetxt(file5,np.transpose(corr_means),delimiter=';')    
np.savetxt(file6,np.transpose(lowers),delimiter=';')    
np.savetxt(file7,np.transpose(uppers),delimiter=';')     
    

dir7 = dir6 + 'permutation_results\\'
os.makedirs(dir7,exist_ok=True)

dir8 = dir6 + 'permutation_results_mop\\'
os.makedirs(dir8,exist_ok=True)

for g in range(N_gr):
    for f in range(N_freq):
        file8 = dir7 + names[g] + ' ' + freq_strings[f] + '.csv'
        np.savetxt(file8,corrsP[f,g],delimiter=';')
    file9 = dir8 + names[g] + '.csv'
    np.savetxt(file9,np.nanmean(corrsP,2)[:,g],delimiter=';')


#  Dimensions:
    
    # All corrs is in shape:  frequency x parcel
    # Mean corrs is in shape: frequency x cohort
    # Grand mean and CLs are in shape:  frequency band x cohort
    # permutation results are in shape: parcel x permutation
    # permutation results mop are in shape: frequency x permutation
