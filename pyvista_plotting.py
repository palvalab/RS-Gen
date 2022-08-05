# -*- coding: utf-8 -*-
"""
@author: Vlad, Felix

Last modified 06.05.2021

"""

import os
import abc
import numpy as np
import pyvista as pv
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

source_directory    = 'K:\\palva\\felix\\'
import sys
sys.path.append(source_directory)
sys.path.append(source_directory + 'Python37\\Utilities')
import parc_functions as parcfun




#%% 

def array_scalar(arr):
    if type(arr) is np.ndarray:
        return np.isscalar(next(iter(arr)))
    else:
        return False

def get_face_label(labels):
    n_parcels = len(set(labels))
    res = labels[0] if (n_parcels == 1) else -1
    return res

def get_triangle_stats(vertex_stats, triangles, func=np.mean):
    res = np.zeros(len(triangles), dtype=vertex_stats.dtype)
    for face_idx, face_stats in enumerate(vertex_stats[triangles]):
        res[face_idx] = func(face_stats)
    return res

class BrainSurface:
    def __init__(self, subject_path, parcellation='Schaefer2018_100Parcels_17Networks', hemis=None, surface='pial',
                  anat_colors = [(0.95,0.95,0.95),(0.82,0.82,0.82)]):
        self.subject_path = subject_path
        self.parcellation = parcellation
        self.surface = surface
        self.anat_colors = anat_colors
        if hemis is None:
            self.hemis = ['lh', 'rh']
        else:
            self.hemis = list(hemis)
        self._load_hemis()
        self._load_annotations()
        self.data = dict()
        self.plotter = None
        
    def _load_hemis(self):
        self.surfaces = dict()
        index_offset = 0
        coords = list()
        self.triangles = list()
        curviture = list()
        for hemi in self.hemis:
            surf_path = os.path.join(self.subject_path, 'surf', f'{hemi}.{self.surface}')
            surf_coords, surf_triangles = nib.freesurfer.io.read_geometry(surf_path)
            surf_triangles += index_offset
            surf_triangles = np.hstack([np.full_like(surf_triangles[:,:1], 3), surf_triangles])
            curv_path = os.path.join(self.subject_path, 'surf', f'{hemi}.curv')
            surf_curv = nib.freesurfer.read_morph_data(curv_path)
            coords.append(surf_coords)
            self.triangles.append(surf_triangles)
            curviture.append(surf_curv)
            index_offset += surf_coords.shape[0]
        coords = np.vstack(coords)
        self.triangles = np.vstack(self.triangles)
        self.vertex_curviture = np.hstack(curviture)
        self.face_curviture = get_triangle_stats(self.vertex_curviture, self.triangles[:, 1:])
        self.brain_mesh = pv.PolyData(coords, self.triangles)
        
    def _load_annotations(self):
        self.annotations = dict()
        self.parcel_names = list()
        vertex_labels = list()
        for hemi in self.hemis:
            annot_path = os.path.join(self.subject_path, 'label', f'{hemi}.{self.parcellation}.annot')
            labels_orig, _, annot_ch_names = nib.freesurfer.io.read_annot(annot_path)   
            labels_orig += len(self.parcel_names)
            annot_ch_names = [n.decode() for n in annot_ch_names]
            vertex_labels += labels_orig.tolist()
            self.parcel_names += annot_ch_names
        self.vertex_labels = np.array(vertex_labels)
        self.face_labels = get_triangle_stats(self.vertex_labels, self.triangles[:, 1:], func=get_face_label)
        
    def _is_scalar(self, data):
        return np.isscalar(next(iter(data.values())))
    
    def set_data(self, data):
        self.data = {key:value for (key,value) in data.items() if key in self.parcel_names}
        
    def plot(self, colormap='viridis', camera_position=None, zoom=1.0, show=True):
        is_scalar = self._is_scalar(self.data)
        scalars = np.full(len(self.face_labels), np.nan)
        colormap = 'viridis' if is_scalar else list()
        for item_index, (key, value) in enumerate(self.data.items()):
            label = self.parcel_names.index(key)
            mask = (self.face_labels == label)
            if is_scalar:
                scalars[mask] = value
            else:
                scalars[mask] = item_index
                colormap.append(value)
        curviture_bin = (self.face_curviture > 0)
        no_data_mask = np.isnan(scalars)
        scalars[no_data_mask] = curviture_bin[no_data_mask] + len(self.data)
        if no_data_mask.sum() > 0:
            colormap.append(self.anat_colors[0])
            colormap.append(self.anat_colors[1])
        if not(is_scalar):
            colormap = mpl.colors.ListedColormap(colormap)
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.add_mesh(self.brain_mesh, scalars=scalars, cmap=colormap, categories=not(is_scalar))
        if not(camera_position is None):
            self.plotter.camera_position = camera_position
        self.plotter.set_background('white')
        self.plotter.camera.Zoom(zoom)
        if show:
            self.plotter.show()
        
    def save_to_image(self, img_path, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        ax.imshow(self.plotter.image)
        ax.set_axis_off()
        fig.savefig(img_path, **kwargs)
        plt.close(fig)
            
            
#%%

def pyvista_multi(left_dict,right_dict,subj_path,parc,figsize=[12,12],filename=None,cmap=None,
                  vrange=None,nticks=11,title=None,
                  hemis  = ['lh','lh','rh','rh'],
                  cpos   = [(-1,0,0),(1,0,0),(1,0,0),(-1,0,0)], 
                  zooms  = [1.75,1.65,1.75,1.65],
                  ):
        
    '''
    If cmap != None, a colorbar is added at the bottom
    '''
    
    npl = len(hemis)
    
    dict_list = []
    for h in hemis:
        if h == 'lh':
            dict_list.append(left_dict)
        else:
            dict_list.append(right_dict)
            
            
    brains = []    
    
    for i in range(npl):
        b = BrainSurface(subj_path,parcellation=parc.name,hemis=[hemis[i]],surface='inflated')  
        b.set_data(dict_list[i])
        b.plot(camera_position=cpos[i],zoom=zooms[i],show=False)
        brains.append(b)
        

    nc = len(hemis)//2

    fig, axes = plt.subplots(2,nc,figsize=figsize)
    for i in range(nc):
        for j in range(2):
            k = j*2+i
            ax = axes[j,i]
            px = ax.imshow(brains[k].plotter.image)
            ax.set_axis_off()
    plt.tight_layout()
    
    if cmap != None:
        fig.subplots_adjust(bottom=0.1)
        axc = fig.add_axes([0, 0, 1, 1])
        tick_arr = np.linspace(vrange[0],vrange[1],nticks,endpoint=True)
        px = axc.imshow(np.array([vrange]),cmap=cmap)
        axc.set_visible(False)
        cbar = plt.colorbar(px,orientation ='horizontal',aspect=36,ticks=tick_arr)
        cbar.ax.tick_params(labelsize=figsize[0]*2)
    plt.suptitle(title,fontsize=figsize[0]*3.5)
    plt.show()

    if filename != None:   
        fig.savefig(filename)
    plt.close(fig)

#%% 

def pyvista_views(left_dict,right_dict,subj_path,parc,figsize,views=['lat-med'],filename=None,
                  cmap=None,vrange=None,nticks=11,title=None):
    
    ''' Creates a 4-brain figure for each entry in "views" (can be 'lat-med','ant_post','top-down').
        Saves figure(s) if a filename is given. Filename must end in a legal file format.
        If "views" is not just 'lat-med', view description is added to filename for each figure.
    
    ''' 
    if filename != None and views != ['lat-med']:
        filenames = {'lat-med' : filename[:-4] + '_lat-med'  + filename[-4:],                     
                     'ant-post': filename[:-4] + '_ant-post' + filename[-4:],
                     'top-down': filename[:-4] + '_top-down' + filename[-4:]}                    
    else:
        filenames = {'lat-med': filename,'ant-post':None,'top-down':None}
            
    
        
    if 'lat-med' in views:
        pyvista_multi(left_dict,right_dict,subj_path,parc,figsize=figsize,cmap=cmap,vrange=vrange,nticks=nticks,
                      hemis  = ['lh','rh','lh','rh'],
                      cpos   = [(-1,0,0),(1,0,0),(1,0,0),(-1,0,0)], 
                      zooms  = [1.7,1.7,1.6,1.6],
                      filename = filenames['lat-med'],
                      title = title
                      )
        
    if 'ant-post' in views:
        pyvista_multi(left_dict,right_dict,subj_path,parc,figsize=figsize,cmap=cmap,vrange=vrange,nticks=nticks,
                      hemis  = ['rh','lh','lh','rh'],
                      cpos   = [(0,1,0),(0,1,0),(0,-1,0),(0,-1,0)], 
                      zooms  = [1.7]*8,
                      filename = filenames['ant-post'],
                      title = title
                      )
        
    if 'top-down' in views:    
        pyvista_multi(left_dict,right_dict,subj_path,parc,figsize=figsize,cmap=cmap,vrange=vrange,nticks=nticks,
                      hemis  = ['lh','lh','rh','rh'],
                      cpos   = [(0,0,1),(0,0,-1),(0,0,1),(0,0,-1)], 
                      zooms  = [1.6]*4,
                      filename = filenames['top-down'],
                      title = title
                      )



#%%

def plot_multi_view_pyvista(data1,parc,colors,subj_path = 'L:\\nttk_palva\\Utilities\\fsaverage\\',
                            filename=None,figsize=[12,12],vrange=None,title=None,
                            views = ['lat-med'],
                            threshold=0,nticks=11):
    ''' 
    INPUT:  
        data1 :   Input data. Can be either:
                    - value array of size N, where N is the number of parcels. colors are picked on a linear scale from 
                            colormap provided in colors. Parcels with value below threshold will not be shown. 
                            A colorbar is added.
                    - color array of size [N x 3]. Parcels with color (-1,-1,-1) not shown.
                    - string 'sys', in which case parcels are colored by subsystem with subsys colors are given in colors
        
        parc:     An instance of the class Parc from parc_functions module.
        colors:   Either: 
                   - a colormap object, or name of a predefined colormap if data is given as a value array
                   - ignored when data1 is an array of colors 
                   - list of N colors for N functional systems when data1 == 'sys'
        filename: If not None, figure(s) will be saved in the file format specified by the filename.
        views:    List of views, can contain 'lat-med', 'top-down' and 'ant-post'. Each view produces a figure with 4 plots.
        threshold: minimum value that is displayed. color range starts from max(threshold, minimum value).
        nticks:   how many ticks in colorbar, if shown.
        '''
        
    left_dict  = {}
    right_dict = {}
    
    if array_scalar(data1):
        if type(colors) not in [mpl.colors.ListedColormap,mpl.colors.LinearSegmentedColormap]:
            cmap =  cm.get_cmap(colors)  
        else:
            cmap = colors
        colors = [cmap(i) for i in range(256)]
        if vrange == None:            
            min1   = np.max([np.min(data1),threshold])
            max1   = np.nanmax(data1)
            vrange = [min1,max1]
        else:
            min1,max1 = vrange
        dx     = 255/(max1 - min1)     
    else:               
        cmap=None 
      
    for p in range(parc.N): 
        name = parc.names[p]
        if name[-6:] == '__Left':
            name = name[:-6] + '-lh'
        if name[-7:] == '__Right':
            name = name[:-7] + '-rh'
            
        if array_scalar(data1):                                # map values to colormap
            if data1[p] >= threshold:         
                ind    = int(((data1[p] - min1) * dx))
                ind    = sorted((0,ind,255))[1]
                color  = colors[ind]
                if name[-2:] == 'lh':               
                    left_dict[name[:-3]]  = color
                else:
                    right_dict[name[:-3]] = color    
                       
        elif data1 == 'sys':                                   # plot systems
            network = parc.networks[p]
            color   = colors[network]                        
            if name[-2:] == 'lh':               
                left_dict[name[:-3]]  = color
            else:
                right_dict[name[:-3]] = color
                
                
        elif len(data1[0]) == 3:                               # plot given colors
            if not data1[p][0] == -1:
                if name[-2:] == 'lh':               
                    left_dict[name[:-3]]  = data1[p]
                else:
                    right_dict[name[:-3]] = data1[p]

                
    pyvista_views(left_dict,right_dict,subj_path,parc,figsize,views,filename,cmap,vrange,nticks,title)
    

#%%

def plot_4_view_with_double_colorscale(data1,data2,parc,colorM,filename=None,border_col=(1,1,1,1),                               
                                subj_path = 'L:\\nttk_palva\\Utilities\\fsaverage\\', 
                                views = ['lat-med'],
                                ):  
    N_sh = len(colorM)
    color1 = np.zeros([N_sh,3])
    color2 = np.zeros([N_sh,3])
      
    left_dict  = {}
    right_dict = {}
    
    # assign color to each parcel
    for i in range(parc.N): 
        ind1 = 0
        ind2 = 0
        if data1[i]>0:
            ind1 = int(data1[i]/np.max(data1)*0.99*N_sh)
        if data2[i]>0:
            ind2 = int(data2[i]/np.max(data2)*0.99*N_sh)
        if data1[i]>0 or data2[i]>0:
            name = parc.names[i]
            if name[-2:] == 'lh':
                left_dict[name[:-3]]  = colorM[ind1][ind2]
            else:
                right_dict[name[:-3]] = colorM[ind1][ind2]
       

    dicts = [left_dict,left_dict,right_dict,right_dict]
    hemis = ['lh','lh','rh','rh']
    
     
    cpos   = [(-1,0,0),(1,0,0),(1,0,0),(-1,0,0)]
    brains = []    
    zooms  = [1.75,1.65,1.75,1.65]
    for i in range(4):
        b = BrainSurface(subj_path,parcellation=parc.name,hemis=[hemis[i]],surface='inflated')  
        b.set_data(dicts[i])
        b.plot(camera_position=cpos[i],zoom=zooms[i],show=False)
        brains.append(b)

        
    fig, axes = plt.subplots(2,2,figsize=[12,12])
    for i in range(2):
        for j in range(2):
            k = i*2+j
            ax = axes[j,i]
            ax.imshow(brains[k].plotter.image)
            ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    if filename != None:   
        fig.savefig(filename)
    plt.close(fig)
   
        
        
  #%% 
def plot_4_view_with_double_colorscale2(data1,data2,parc,colors=['Reds','Blues'],N_shades=100,                                
                                subj_path = 'L:\\nttk_palva\\Utilities\\fsaverage\\'):  
    N_sh = N_shades
    color1 = np.zeros([N_sh,3])
    color2 = np.zeros([N_sh,3])
    
    cmap = mpl.cm.get_cmap(colors[0])
    for i in range(N_sh):
            rgba = np.around(cmap(i*(1/N_sh)),3)
            color1[i] = rgba[:3]     
    
    cmap = mpl.cm.get_cmap(colors[1])
    for i in range(N_sh):
            rgba = np.around(cmap(i*(1/N_sh)),3)
            color2[i] = rgba[:3]  
            
            
            
    max_1  = np.max(data1)
    max_2  = np.max(data2)
    N_parc = len(data1)
    
    left_dict  = {}
    right_dict = {}
    
    # assign color to each parcel
    for i in range(parc.N):
    
      try:
        if data1[i] !=0 and data2[i]==0:
            ind = np.int(np.ceil(N_sh*(data1[i]/max_1)))-1
            colors = color1[ind]
        if data2[i] !=0 and data1[i]==0:
            ind = np.int(np.ceil(N_sh*(data2[i]/max_2)))-1
            colors = color2[ind]
        if data2[i] !=0 and data1[i]!=0:
            ind1 = np.int(np.ceil(N_sh*(data1[i]/max_1)))-1
            ind2 = np.int(np.ceil(N_sh*(data2[i]/max_2)))-1
            colors = (color1[ind1]*ind1 + color2[ind2]*ind2)/(ind1+ind2)
        if data1[i] !=0 or data2[i] !=0: 
            name = parc.names[i]
            if name[-2:] == 'lh':
                left_dict[name[:-3]]  = colors
            else:
                right_dict[name[:-3]] = colors
      except:
          print(ind)
            

    dicts = [left_dict,left_dict,right_dict,right_dict]
    hemis = ['lh','lh','rh','rh']
    cpos  = [(-1,0,0),(1,0,0),(-1,0,0),(1,0,0)]
    
    for i in range(4):
        b = pyvi.BrainSurface(subj_path,parcellation='parc2009',hemis=[hemis[i]],surface='inflated')  
        b.set_data(dicts[i])
        b.plot(camera_position=cpos[i]) 