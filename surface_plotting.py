# -*- coding: utf-8 -*-
"""

Created: 13.08.2018
@author: Felix Siebenhühner
Tested with Python 3, mne 0.20 and pysurfer 0.10
Recommended to use QT as GUI in the calling script, otherwise Python may crash!

"""


from surfer import Brain
import nibabel as nib
import numpy as np
import pyvista as pv
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import parc_functions as parcfun
import pyvista_plotting as pyvi

#%% load parcels from label files

def load_parcels(sub_dir,subj,parcellation):
    if parcellation == 'parc2009':                
        aparc_lh_file = sub_dir + '\\' + subj + '\\label\\lh.aparc.a2009s.annot'
        aparc_rh_file = sub_dir + '\\' + subj + '\\label\\rh.aparc.a2009s.annot'
    else:
        aparc_lh_file = sub_dir + '\\' + subj + '\\label\\lh.' + parcellation + '.annot'  
        aparc_rh_file = sub_dir + '\\' + subj + '\\label\\rh.' + parcellation + '.annot' 
        
    labels_lh, ctab, names_lh = nib.freesurfer.read_annot(aparc_lh_file)
    labels_rh, ctab, names_rh = nib.freesurfer.read_annot(aparc_rh_file) 
    
    names_lh  = [str(n)[2:-1] +'-lh' for n in names_lh]
    names_rh  = [str(n)[2:-1] + '-rh' for n in names_rh] 

    return names_lh, names_rh, labels_lh, labels_rh

#%% plot with pysurfer

def plot_4_view(data1,parcel_names,parcellation,
                style='linear',alpha=1,thresh=-100,figsize=[900,800],
                zmin=None,zmax=None,zmid=None,cmap='auto',show=True,
                filename=None,surface='inflated',null_val=0,
                transparent = True,subj='fsaverage - old', title = '',               
                sub_dir='L:\\nttk_palva\\Utilities\\'):
    
    '''         
    Plots 1d array of data. Plotted views are lateral and medial on both HS.
    Used brain is fsaverage.
    
    INPUT:            
        data1:        1-dimensional data array, len = # parcels. 
                      1st half must be left HS, 2nd half right.
        parcel_names: Parcel_names, in the same order as the data.                       
        parcellation: Abbreviation, e.g. 'parc2018yeo7_100' or "parc2009'
        style:        'linear': pos. values only, 'divergent': both pos & neg
        alpha:        Transparency value; transparency might look weird.
        zmin:         The minimum value of a linear z-axis, or center of a 
                        divergent axis (probably 0). Can't be negative!
        zmax:         Maximum value of linear z-axis, or max/-min of div.               
        zmid:         Midpoint of z-axis.
        cmap:         Colormap by name. Default is 'rocket' for linear, and
                        'icefire' for divergent; other recommended options: 
                         'YlOrRd' for linear,  or 'bwr' for divergent.
        show:         If False, plot is closed after creation. 
        filename:     File to save plot as, e.g. 'plot_13.png'
        surface:      Surface type.
        null_val:     Value for unassigned vertices
        thresh:       values below this treshold are not shown.
        transparent:  Whether parcels with minimum value should be transparent.
        
    OUTPUT:
        instance of surfer.Brain, if show==True
    '''
    
    N_parc = len(data1)    # the number of actually used parcels
    if len(parcel_names) != N_parc:
        raise ValueError('Number of parcels != len(data1) ')
    
    for i in range(N_parc):
        p = parcel_names[i]
        p = p.replace('__Left','')
        p = p.replace('__Right','')        
        parcel_names[i]=p

    if parcel_names[0][-3:] != '-lh':
       parcel_names[:N_parc//2] = [p + '-lh' for p in parcel_names[:N_parc//2]]
       parcel_names[N_parc//2:] = [p + '-rh' for p in parcel_names[N_parc//2:]]

    
    hemi = 'split'
       
    names_lh, names_rh, labels_lh, labels_rh = load_parcels(sub_dir,subj,parcellation)
    
    N_label_lh   = len(names_lh)      # number of labels/parcels with unkown and med. wall included
    N_label_rh   = len(names_rh) 

    #### map parcels in data to loaded parcels
    indicesL = np.full(N_label_lh,-1)
    indicesR = np.full(N_label_rh,-1)
    
    for i in range(N_parc):
        for j in range(N_label_lh):
            if names_lh[j]==parcel_names[i]:
                indicesL[j]=i 
        for j in range(N_label_rh):
            if names_rh[j]==parcel_names[i]:            
                indicesR[j]=i-N_parc//2     
    indicesL += 1
    indicesR += 1

    
    ## assign values to loaded parcels
    data1L     = np.concatenate(([null_val],data1[:N_parc//2]))
    data1R     = np.concatenate(([null_val],data1[N_parc//2:]))
    data_left  = data1L[indicesL]
    data_right = data1R[indicesR]
    
    ## map parcel values to vertices 
    vtx_data_left = data_left[labels_lh]
    vtx_data_left[labels_lh == -1] = null_val
    vtx_data_right = data_right[labels_rh]
    vtx_data_right[labels_rh == -1] = null_val

    
    if zmin == None:
        zmin = 0
    if zmax == None:
        zmax = np.nanmax(abs(data1))
        if zmax == 0 == zmin:
            zmax = 1
    if zmid == None:
        zmid = (zmax-zmin)/2+zmin

    
    if style == 'linear':           # shows only positive values 
        center = None
    elif style == 'divergent':      # shows positive and negative values
        center =  0
    
       
    #### plot to 4-view Brain
    hemi = 'split'
    brain = Brain(subj, hemi, background = 'white', surf = surface, 
                  size = figsize, title = title, cortex = 'classic',
                   subjects_dir=sub_dir,  views = ['lat', 'med']) 
    try:
        brain.add_data(vtx_data_left,  zmin, zmax, colormap=cmap, thresh=thresh,
                   center= center, alpha=alpha, hemi='lh')
        brain.add_data(vtx_data_right, zmin, zmax, colormap=cmap, thresh=thresh,
                   center= center, alpha=alpha, hemi='rh')
    except:
        brain.add_data(vtx_data_left,  zmin, zmax, colormap=cmap, thresh=thresh,
                    alpha=alpha, hemi='lh')
        brain.add_data(vtx_data_right, zmin, zmax, colormap=cmap, thresh=thresh,
                    alpha=alpha, hemi='rh') 
        
        
    # adjust colorbar
    brain.scale_data_colormap(zmin, zmid, zmax, transparent=transparent, 
                              center=center, alpha=alpha, data=None, hemi=None,
                              verbose=None)


    if filename != None:
        brain.save_image(filename) 
    
    if show:
        return brain
    
#%% 

def find_7max(value,round='up'):
    """ a funny little function to increase a value to a nearby 
    value neatly divisible by 7, useful for choosing zmax for pysurfer """
    
    pot    = np.floor(np.log10(value))-1
    if round == 'up':
        value1 = np.ceil(value / 10**pot)
    else:
        value1 = np.floor(value / 10**pot)
    while value1%7 != 0:
        value1 +=1
    return value1 * 10**pot
    
    
    
    