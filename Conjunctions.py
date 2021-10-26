# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:55:22 2021
@author: Felix Siebenhühner
notes: python 3.7
"""

source_directory    = 'code_location\\'
project_dir         = 'main_directory\\'

import sys
sys.path.append(source_directory)

import numpy as np
import parc_functions as parcfun
import plot_functions as plots
import pyvista_plotting as pyv
import os


#%% basic settings

parc = parcfun.Parc('parc2009')

ddir = project_dir + 'Data\\Plot_data\\_per_band\\'

suffix = '_sig_any'

dtypes  = ['Amplitude', 'DFA']
pair_names = ['COMT ' + p for p in ['Val-Met - Met-Met', 'Val-Met - Val-Val', 'Val-Val - Met-Met']
                ] + ['BDNF Val - Met']

pair_names_inv = ['COMT ' + p for p in ['Met-Met - Val-Met', 'Val-Val - Val-Met', 'Met-Met - Val-Val']
                ] + ['BDNF Met - Val']

f_bands    = ['3 - 7 Hz', '7 - 18 Hz', '20 - 60 Hz']

#%% selections in [ (dtype1, dtype2), (pair1, pair2), f_band, (invert1, invert2) ]

selections = [
       
    [(0,0), (0,3), 0, (0,0)],           
    [(0,0), (1,3), 0, (0,0)],           
    [(0,0), (0,3), 2, (1,1)],           
    [(0,0), (1,3), 2, (1,1)],           
    [(0,0), (0,3), 1, (0,0)],   
   
    [(1,1), (0,3), 0, (0,0)],           
    [(1,1), (2,3), 0, (0,0)],    
    [(1,1), (0,0), 2, (0,2)],
    [(1,1), (1,1), 2, (0,2)],
    [(1,1), (3,3), 1, (2,0)],
    
    [(0,1), (0,0), 0, (0,0)],           
    [(0,1), (0,0), 2, (0,0)],           
    [(0,1), (1,1), 2, (0,0)],           
    [(0,1), (3,3), 0, (0,0)],
    ]






#%%

hc = ['b','g','c','y','b','g','c','y','b','g','c','y']

pctile    = 0

fig, colorM = plots.make_double_colorscale_red_blue(N_shades=50,return_map=1)

outdir  = project_dir + '\\Plots\\Conjunction plots any sig\\'
os.makedirs(outdir,exist_ok=True)

export = 0

for s in range(0,14):
   
    sel = selections[s]
    
    dt1 = dtypes[sel[0][0]]
    dt2 = dtypes[sel[0][1]]
    pn1 = pair_names[sel[1][0]]
    pn2 = pair_names[sel[1][1]]
    
    file1 = ddir + dt1 + ' diffs' + suffix + '\\' + pn1 + ', ' + f_bands[sel[2]] + '.csv'
    file2 = ddir + dt2 + ' diffs' + suffix + '\\' + pn2 + ', ' + f_bands[sel[2]] + '.csv'
    
    data1 = np.genfromtxt(file1)
    data2 = np.genfromtxt(file2)
    
    if sel[3][0] == 1:
        data1 = -1 * data1 
        pn1   = pair_names_inv[sel[1][0]]
    elif sel[3][0] == 2:
        data1 = 0 * data1
        dt1 = pn1 = ''        
    if sel[3][1] == 1:
        data2 = -1 * data2
        pn2   = pair_names_inv[sel[1][1]]
    elif sel[3][1] == 2:
        data2 = 0 * data2
        dt2 = pn2 = '' 
        
    
    title   = dt1 + ' ' + pn1 + ', ' + f_bands[sel[2]] \
    + ' vs. ' + dt2 + ' ' + pn2 + ', ' + f_bands[sel[2]]
    
    print(title)
    
    data1 = data1 * (data1 > np.percentile(data1,pctile))  * (data1 > 0)
    data2 = data2 * (data2 > np.percentile(data2,pctile))  * (data2 > 0)
        
    if export:
        filename =  outdir + str(s+1) + ' ' + title + '.eps'
    else:
        filename = None
    
    pyv.plot_4_view_with_double_colorscale(data1,data2,parc,colorM,
                                    filename=filename)
    
    



