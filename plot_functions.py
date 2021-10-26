# -*- coding: utf-8 -*-
"""
@author: Felix Siebenhühner
"""

import numpy as np
import matplotlib as mpl
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
from numpy.ma import masked_invalid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import mpl_toolkits
import enum

''' list of functions: 
    
    make_cmap: creates colormap from list of RGB values
    
    plot_lines: simple line plot with easy customization
    
    plot_heatmap: plots 2D data as a "heatmap"
    
    plot_heatmaps: plots multiple heatmaps into one figure
    
    semi_log_plot: plots data series over log-spaced frequencies 
                   optionally with confidence intervals/standard deviation
                   
    semi_log_plot_multi: plots data series over log-freqs. in subplots 
                   optionally with confidence intervals/standard deviation
                    
    simple_CF_plot: heatmap plot specifically for the visualization of 
                multi-ratio cross-frequency interactions
                    
'''
class Bunch():
    __init__ = lambda self, **kw: setattr(self, '__dict__', kw)
    
def set_mpl_defs():
    mpl.style.use('default')
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['font.size'] =  12
    mpl.rcParams['axes.titlesize'] =  12
    mpl.rcParams['axes.labelsize'] =  12
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['xtick.labelsize'] =  11
    mpl.rcParams['ytick.labelsize'] =  11
    mpl.rcParams['lines.linewidth'] =  1
    mpl.rcParams['axes.facecolor'] = '1'
    mpl.rcParams['figure.facecolor'] = '1'

def set_mpl_defs2():
    mpl.style.use('default')
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['font.size'] =  15
    mpl.rcParams['axes.titlesize'] =  18
    mpl.rcParams['axes.labelsize'] =  15
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['xtick.labelsize'] =  13
    mpl.rcParams['ytick.labelsize'] =  13
    mpl.rcParams['lines.linewidth'] =  1
    mpl.rcParams['axes.facecolor'] = '1'
    mpl.rcParams['figure.facecolor'] = '1'

def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            print("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            print("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap

def _depth(obj,c=0):
     try:
         if type(obj) != str:
             obj1 = obj[0]
             c = _depth(obj1,c+1)
         else:
             return c
     except:
         return c
     return c



def mix_cmap(n,repeat=1,greys=0):
    mixed_colors = [(1,0,0),(0,.9,0),(0,0,1),(.6,0.,9),(.8,.6,0),(0,.9,.9)]
    colors = mixed_colors[:n]*repeat+[(.6,.6,.6)]*greys
    return make_cmap(colors)

def def_color_arrays():
    
    mca = Bunch()
    
    mca.brgg  = [(0, 0, 1), (1, 0, 0), (.4,.4,.4), (.4,.4,.4)]
    mca.rbgg  = [(1, 0, 0), (0, 0, 1), (.4,.4,.4), (.4,.4,.4)]
    
    mca.yeo7  = [(0.9, 0.3, 0.0),  # FP/control
                 (1.0, 0.0, 0.0),  # DMN
                 (0.0, 1.0, 0.0),  # Dorsal Attention (DAN)
                 (0.4, 0.4, 0.4),  # Limbic            
                 (1.0, 0.3, 0.6),  # Salience & Ventral Attention (VAN)
                 (0.0, 0.0, 1.0),  # Somatomotor (SM)
                 (0.6, 0.0, 0.8)]  # Visual
    
    mca.yeo7a = [(1.0, 0.6, 0.2),  # FP/control
                 (1.0, 0.0, 0.0),  # DMN
                 (0.0, 1.0, 0.0),  # Dorsal Attention (DAN)
                 (1.0, 0.9, 0.8),  # Limbic
                 (1.0, 0.5, 0.8),  # Salience & Ventral Attention (VAN)
                 (0.0, 0.0, 1.0),  # Somatomotor (SM)
                 (0.6, 0.0, 0.8)]  # Vis
    
    mca.yeo7b = [(1.0, 0.7, 0.3),  # FP/control
                 (1.0, 0.1, 0.1),  # DMN
                 (0.1, 1.0, 0.1),  # Dorsal Attention (DAN)
                 (1.0, 0.9, 0.8),  # Limbic
                 (1.0, 0.6, 0.8),  # Salience & Ventral Attention (VAN)
                 (0.1, 0.1, 1.0),  # Somatomotor (SM)
                 (0.7, 0.0, 0.9)]  # Vis
             
    mca.yeo17 = [(1.0, 0.7, 0.0),  # Control A
                 (0.9, 0.6, 0.0),  # Control B
                 (0.8, 0.5, 0.2),  # Control C
                 (1.0, 0.0, 0.0),  # Default A 
                 (0.9, 0.2, 0.2),  # Default B 
                 (0.8, 0.0, 0.0),  # Default C
                 (0.0, 0.9, 0.0),  # Dorsal Attention A
                 (0.0, 0.7, 0.0),  # Dorsal Attention B  
                 (0.6, 0.6, 0.6),  # Limbic OFC 
                 (0.3, 0.3, 0.3),  # Limbic Temp. Pole
                 (1.0, 0.3, 0.6),  # Sal/Ventral Attention A
                 (0.8, 0.2, 0.7),  # Sal/Ventral Attention B
                 (0.3, 0.3, 1.0),  # Somatomotor A
                 (0.0, 0.0, 0.7),  # Somatomotor B
                 (0.0, 0.0, 0.0),  # Temporal-parietal
                 (0.7, 0.0, 0.8),  # Visual Central
                 (0.3, 0.0, 0.5)]  # Visual Peripheral
                    
    mca.blues  = [(0,0,0.4),  (0,0,0.7), (0,0,1), (0,0.3,1),(0.2,0.4,1)]
    mca.reds   = [(0.4,0,0),  (0.7,0,0), (1,0,0), (1,0.3,0),(1,0.4,.2)]
    mca.greens = [(0,0.4,0),  (0,0.7,0), (0,1,0), (0,1,0.3),(0.2,1,0.4)]
    mca.bggr  = [(0,0,1), (0,1,0), (0,1,0), (1,0,0)]
    mca.rggb  = [(1,0,0), (0,1,0), (0,1,0), (0,0,1)]
    mca.rgbb  = [(1,0,0), (0,0.6,0), (0,0,0), (0,0,1)]   

    return mca              


def make_cmaps():
    
    my_cmaps = Bunch()
    mca      = def_color_arrays()
       
    my_cmaps.blues    = make_cmap(mca.blues)
    my_cmaps.reds     = make_cmap(mca.reds)
    my_cmaps.greens   = make_cmap(mca.greens)   
    my_cmaps.bggr     = make_cmap(mca.bggr)
    my_cmaps.rggb     = make_cmap(mca.rggb)
    my_cmaps.rgbb     = make_cmap(mca.rgbb)
    my_cmaps.brgg     = make_cmap(mca.brgg)
    my_cmaps.rbgg     = make_cmap(mca.rbgg)
        
    my_cmaps.yeo7     = make_cmap(mca.yeo7)
    my_cmaps.yeo7a    = make_cmap(mca.yeo7a)
    my_cmaps.yeo7e    = make_cmap([(0.8,0.8,0.8)]+mca.yeo7)
    my_cmaps.yeo7_2x  = make_cmap(mca.yeo7 + mca.yeo7)
    
    my_cmaps.yeo17    = make_cmap(mca.yeo17)
    my_cmaps.yeo17b   = make_cmap([(0.8,0.8,0.8)]+mca.yeo17)
    my_cmaps.yeo17_2x = make_cmap(mca.yeo17 + mca.yeo17)
    

 
    return my_cmaps

#%%

def make_double_colorscale(colors=['Reds','Blues'],N_shades=100,
                           figsize=[7,7],return_map=False):
    N_sh = N_shades
    color_A = np.zeros([N_sh,4])
    color_B = np.zeros([N_sh,4])
    
    cmap = mpl.cm.get_cmap(colors[0])
    for i in range(N_sh):
            rgba = np.around(cmap(i*(1/N_sh)),3)
            color_A[i] = (rgba) 
    
    cmap = mpl.cm.get_cmap(colors[1])
    for i in range(N_sh):
            rgba = np.around(cmap(i*(1/N_sh)),3)
            color_B[i] = (rgba) 
         
    ind = 0
    colors = np.zeros([N_sh**2,4])
    values = np.zeros([N_sh,N_sh])
    colors_map = [[[] for i in range(N_sh) ] for j in range(N_sh)]
        
    for i in range(N_sh):
        for j in range(N_sh):
            if i==0:
                colors[ind] = color_B[j]
            elif j==0:
                colors[ind] = color_A[i]
            else:
                colors[ind] = (color_A[i]*i + j*color_B[j])/(i+j)
           # colors[ind][1] =(1-i/N_sh)*(1-j/N_sh)
            values[i,j] = ind
            colors_map[i][j] = colors[ind]
            ind +=1
                
    u = np.unique(values)
    bounds = np.concatenate(([values.min()-1], u[:-1]+np.diff(u)/2. ,[values.max()+1]))
    norm = mpl.colors.BoundaryNorm(bounds, len(bounds)-1)
    cmap = mpl.colors.ListedColormap(colors) 
    
    ticks = np.arange(0,N_sh+1,N_sh/5)
    ticklabels = [str(np.around(l,1)) for l in np.arange(0,1.1,0.2)]
    ticks = ticks - 0.5
    
    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.set_xticks(ticks);     
    ax.set_xticklabels(ticklabels);
    ax.set_yticks(ticks); 
    ax.set_yticklabels(ticklabels);
    ax.imshow(values, cmap=cmap,norm=norm,origin='bottom')
    if return_map:
        return fig, colors_map
    else:
        return fig




def make_double_colorscale_red_blue(N_shades=100,figsize=[7,7],return_map=False):
    N_sh = N_shades
    color_A = np.zeros([N_sh,4])
    color_B = np.zeros([N_sh,4])
    
    cmap = mpl.cm.get_cmap('Reds')
    for i in range(N_sh):
            rgba = np.around(cmap(i*(1/N_sh)),3)
            color_A[i] = (rgba) 
    
    for i in range(N_sh):
        color_B[i] = (color_A[i][2]*0.95,color_A[i][1],color_A[i][0]*0.95,color_A[i][3]) 
         
    ind = 0
    colors = np.zeros([N_sh**2,4])
    values = np.zeros([N_sh,N_sh])
    colors_map = [[[] for i in range(N_sh) ] for j in range(N_sh)]
        
    for i in range(N_sh):
        for j in range(N_sh):
            if i==0 and j==0:
                colors[ind] = [1,1,1,1]
            elif i==0:
                colors[ind] = color_B[j]
            elif j==0:
                colors[ind] = color_A[i]            
            else:
                colors[ind] = (color_A[i]*i + j*color_B[j])/(i+j)
            values[i,j] = ind
        #    colors[ind] = (color_A[i]*i + j*color_B[j])/(i+j)
            colors_map[i][j] = colors[ind]
            ind +=1
                
    u = np.unique(values)
    bounds = np.concatenate(([values.min()-1], u[:-1]+np.diff(u)/2. ,[values.max()+1]))
    norm = mpl.colors.BoundaryNorm(bounds, len(bounds)-1)
    cmap = mpl.colors.ListedColormap(colors) 
    
    ticks = np.arange(0,N_sh+1,N_sh/5)
    ticklabels = [str(np.around(l,1)) for l in np.arange(0,1.1,0.2)]
    ticks = ticks - 0.5
    
    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.set_xticks(ticks);     
    ax.set_xticklabels(ticklabels);
    ax.set_yticks(ticks); 
    ax.set_yticklabels(ticklabels);
    ax.imshow(values, cmap=cmap,norm=norm,origin='bottom')
    if return_map:
        return fig, colors_map
    else:
        return fig


#%%    

def plot_lines(data, names=None, figsize=[13,4], cmap='jet', 
               xlabel ='', ylabel = '', xticks = None, yticks = None, 
               xticklabels = None, xticklab_rot = 0, marker=None,
               title = None, less_spines = True, zero_line=False,
               outfile = None, xlim = None, ylim = None, fontsize=12):
    ''' INPUT:
        data:    1D array or list, or 2D array or list
        names:   list of names of data series for legend
        figsize: Figure size
        cmap:    Colormap. Can be:
                 - the name of a library cmap
                 - an instance of mpc.LinearSegmentedColormap    
                 - a list of colors as characters or RGB tuples (in 0-1 range)
        xlabel, ylabel: axis labels
        xticks, yticks: axis ticks, list of int
        less_spines: no axes on right and top 
        outfile: file to save the plot to 
        xlim, ylim: x and y limits, e.g. xlim=[-4,4]
    '''  
    try:                            # if 1D, force to 2D
        d54 = data[0][0]
    except IndexError:
        data = [data]
    
    if type(cmap) is list:
        colors = cmap
    elif type(cmap) is mpc.LinearSegmentedColormap:
        colors = [cmap(i) for i in np.linspace(0, 1, len(data))] 
    else:
        colors = [plt.get_cmap(cmap)(i) for i in np.linspace(0, 1, len(data))]         
                
    fig = plt.figure(figsize=figsize)
    ax  = plt.subplot(111)    
    for i in range(len(data)):
        ax.plot(data[i],color=colors[i],marker=marker)

    ax.tick_params(labelsize=fontsize-2)
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_ylabel(ylabel,fontsize=fontsize)
    if np.all(names != None):
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(names,loc='center left', bbox_to_anchor=(1, 0.5),fontsize=fontsize)

    if np.all(xticks != None):        
        ax.set_xticks(xticks)  
    if np.all(xticklabels != None):
        ax.set_xticklabels(xticklabels,fontsize=fontsize,rotation=xticklab_rot)
    if np.all(yticks != None):        
        ax.set_yticks(yticks)
    if np.all(xlim != None):
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    if title != None:
        fig.suptitle(title)
    if zero_line:
        ax.plot(np.zeros(len(data[0])),color=(0.5,0.5,0.5))
        
    if less_spines:        
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    
    if outfile != None:
        plt.savefig(outfile) 

    

        
    
def plot_heatmap(data,figsize=[9,7],
                 cbar='right',zmax=None,zmin=None,cmap=plt.cm.YlOrRd,  
                 xlabel='',ylabel='',zlabel='',fontsize = 18,
                 xticks=None,yticks=None,zticks=None,showticks=[1,1,0,0],
                 xticklabels=None, yticklabels='none',interpol='none',
                 zticklabels=None, xticklab_rot=45,title=None,aspect=None,cbarf=0.03,
                 masking='False',bad='white',under='None', topaxis=False):
    ''' INPUT:
        data:                 2D array
        figsize:              figure size             
        cbar:                 can be 'right', 'bottom' or None
        zmax:                 max value for z axis (auto if None)
        zmin:                 min value for z axis (epsilon if None)
        cmap:                 colormap for colorbar; either:
                                - Instance of mpc.LinearSegmentedColormap 
                                - name of library cmap                            
        xlabel,ylabel,zlabel: labels for x, y, z axes
        fontsize:             fontsize for major labels
        xticks,yticks,zticks: tick values for x, y, z axes
        showticks:            whether to show ticks on bottom,left,top,right
        xticklabels:          labels for x ticks
        yticklabels:          labels for y ticks
        xticklab_rot:         degrees by which to rotate xticklabels
        interpol:             interpolation, can be e.g. 'antialiased', 'nearest' 
        title:                title for the plot
        aspect:               sets ratio of x and y axes, automatic by default
        cbarf:                regulates size of colorbar
        masking:              whether to mask invalid values (inf,nan)
        bad:                  color for masked values
        under:                color for low out-of range values
        topaxis:              whether to have additional x axis on top
    
    '''
    
    eps = np.spacing(0.0)
    fig, ax = plt.subplots(1,figsize=figsize)
    if aspect == None:
        aspect = data.shape[1]/data.shape[0] * figsize[1]/figsize[0]

    if type(cmap) != mpc.LinearSegmentedColormap:
        cmap = mpl.cm.get_cmap(cmap)
    cmap.set_bad(bad)
    cmap.set_under(under)        
    
    if masking:
        data = masked_invalid(data)
    if topaxis:
        ax.tick_params(labeltop=True)        
    if zmax == None:
        zmax = max(np.reshape(data,-1))  
    else:
        mask1 = data>zmax
        data  = data*(np.invert(mask1)) + zmax*0.999*mask1
    if zmin != None:
        mask2 = data<zmin
        data  = data*(np.invert(mask2)) + zmin*0.999*mask2   
        
    if zmin==None or zmin==0:    
       # PCM = ax.pcolormesh(data,vmin=eps, vmax=zmax,cmap=cmap,shading=shading)   
        PCM = ax.imshow(data,vmin=eps, vmax=zmax,cmap=cmap,aspect=aspect,interpolation=interpol) 
    else:
       # PCM = ax.pcolormesh(data,vmin=zmin,vmax=zmax,cmap=cmap,shading=shading)   
        PCM = ax.imshow(data,vmin=zmin,vmax=zmax,cmap=cmap,aspect=aspect,interpolation=interpol)   

    
    ax.set_xlim([-0.5,len(data[0])-0.5])
    ax.set_ylim([-0.5,len(data)-0.5])
    
 
    
    if np.all(xticks != None): 
        if _depth(xticks) == 0:    
            Nx=len(xticklabels)
            ax.set_xticks(np.arange(Nx)+0.5)
            ax.set_xlim([0,Nx]) 
        else:
            ax.set_xticks(xticks)
        if np.all(xticklabels!=None):
            ax.set_xticklabels(xticklabels,rotation=xticklab_rot)
            
    if np.all(yticks != None): 
        if _depth(yticks) == 0:    
            Ny=len(yticklabels)
            ax.set_yticks(np.arange(Ny)+0.5)
            ax.set_ylim([0,Ny]) 
        else:
            ax.set_yticks(yticks)
        if np.all(yticklabels !='none'):
            ax.set_yticklabels(yticklabels)

        
    ax.set_xlabel(xlabel,fontsize=18)
    ax.set_ylabel(ylabel,fontsize=18)
    ax.tick_params(axis='both',which='both',labelsize=fontsize-2)
    ax.tick_params(bottom=showticks[0], left=showticks[1], top=showticks[2], right=showticks[3])
    if cbar == 'bottom':
        orient = 'horizontal'
    else:
        orient = 'vertical'
    if cbar != None:    
        if zticks !=None:
            cb  = plt.colorbar(PCM, ax=ax, ticks = zticks, orientation = orient,fraction=cbarf)
            if zticklabels  != None:
                if orient == 'vertical':
                    cb.ax.set_yticklabels(zticklabels)
                else:
                    cb.ax.set_xticklabels(zticklabels)
        else:
            cb  = plt.colorbar(PCM, ax=ax, orientation = orient,fraction=cbarf)
        cb.set_label(zlabel,fontsize=18)
        cb.ax.tick_params(labelsize=14) 

    if title != None:
        plt.title(title,fontsize=18)


        
    
def plot_heatmap_with(data,data2,data3,alpha,Q,linecol='k',figsize=[9,7],
                 zmax=None,zmin=None,cmap=plt.cm.YlOrRd,  
                 xlabel='',ylabel='',zlabel='',fontsize = None,
                 xticks=None,yticks=None,zticks=None,
                 xticklabels=None, yticklabels='none',xmax_n=None,
                 zticklabels=None, xticklab_rot=45,title=None,ymax_f=None,
                 masking='False',bad='white',under='None', topaxis=False):
    ''' INPUT:
        data:                 2D array
        figsize:              figure size             
        cbar:                 can be 'right', 'bottom' or None
        zmax:                 max value for z axis (auto if None)
        zmin:                 min value for z axis (epsilon if None)
        cmap:                 colormap for colorbar; either:
                                - Instance of mpc.LinearSegmentedColormap 
                                - name of library cmap                            
        xlabel,ylabel,zlabel: labels for x, y, z axes
        fontsize:             fontsize for major labels
        xticks,yticks,zticks: tick values for x, y, z axes
        xticklabels:          labels for x ticks
        yticklabels:          labels for y ticks
        xticklab_rot:         degrees by which to rotate xticklabels
        masking:              whether to mask invalid values (inf,nan)
        bad:                  color for masked values
        under:                color for low out-of range values
        topaxis:              whether to have additional x axis on top
    
    '''
    
    
    data  = data  - alpha*100
    data2 = data2 - alpha*100
    data3 = data3 - alpha*100

    
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_axes([0.0,  0.27, 0.8, 0.64])
    ax2 = fig.add_axes([0.87, 0.27, 0.1, 0.64])
    ax3 = fig.add_axes([0.0,  0.0,  0.8, 0.15])
    ax4 = fig.add_axes([0.87, 0.06, 0.1, 0.04])
    
       
    # ax  = fig.add_axes([0.2, 0.25, 0.9,  0.7])
    # ax2 = fig.add_axes([0.0, 0.25, 0.1, 0.7])
    # ax3 = fig.add_axes([0.2, 0.0, 0.7, 0.15])
    
    # 
    eps = np.spacing(0.0)
    
    
    if fontsize==None:
        fontsize=figsize[1]+figsize[0]
    
    if cmap == 'Blues':
        linecol = 'b'
    elif cmap == 'Reds':
        linecol = 'r'
        
    if type(cmap) != mpc.LinearSegmentedColormap:
        cmap = mpl.cm.get_cmap(cmap)
    cmap.set_bad(bad)
    cmap.set_under(under)     
    

    
    if masking:
        data = masked_invalid(data)
    if topaxis:
        ax.tick_params(labeltop=True)        
    if zmax == None:
        zmax = max(np.reshape(data,-1))     
    if zmin==None or zmin==0:    
        PCM = ax.pcolormesh(data,vmin=eps,vmax=zmax,cmap=cmap)   
    else:
        PCM = ax.pcolormesh(data,vmin=zmin,vmax=zmax,cmap=cmap)   

    
    ax.set_xlim([0,len(data[0])])
    ax.set_ylim([0,len(data)])
    
    if np.all(xticks != None): 
        if _depth(xticks) == 0:    
            Nx=len(xticklabels)
            ax.set_xticks(np.arange(Nx)+0.5)
            ax.set_xlim([0,Nx]) 
        else:
            ax.set_xticks(xticks)
        if xticklabels !=None:
            ax.set_xticklabels(xticklabels,rotation=xticklab_rot)
            
    if np.all(yticks != None):         
        ax.tick_params(axis='y',left=True,right=True,labelright=True)
        if _depth(yticks) == 0:    
            Ny=len(yticklabels)
            ax.set_yticks(np.arange(Ny)+0.5)
            ax.set_ylim([0,Ny]) 
        else:
            ax.set_yticks(yticks)
        if yticklabels !='none':
            ax.set_yticklabels(yticklabels)

        
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_ylabel(ylabel,fontsize=fontsize)
    ax.tick_params(axis='both',which='both',length=0,labelsize=fontsize-2)

    orient = 'horizontal'

       
    if zticks !=None:
        cb  = plt.colorbar(PCM,cax=ax4,ticks=zticks,orientation = orient)                               
        if zticklabels  != None:
            if orient == 'vertical':
                cb.ax.set_yticklabels(zticklabels)
            else:
                cb.ax.set_xticklabels(zticklabels)
    else:
        cb  = plt.colorbar(PCM, cax=ax4, orientation = orient)
    cb.set_label(zlabel,fontsize=fontsize)
    cb.ax.tick_params(labelsize=fontsize*0.7) 

        
       
    # simple plot of mean K over freqs AFO network
    
    ax2.plot(data2,np.arange(0,len(data2)),linecol,marker='o')
    if xmax_n == None:
        xmax_n = 1+np.ceil(zmax/3)
    ax2.set_xlim([0,xmax_n])
    ax2.set_ylim([-.5,len(data2)-0.5])
    ax2.get_yaxis().set_visible(False)
    ax2.tick_params(labelsize=fontsize-3)

    # simple plot of mean K over parcels AFO freq.   
     
    ax3.plot(data3,linecol,marker='o')
    ax3.plot(np.full(len(data3),(Q)*100),'k')
    ax3.set_xlim([0,len(data3)-1])
    if ymax_f == None:
        ymax_f = 1 + np.ceil(np.max(zmax)/3)
    ax3.set_ylim([0,ymax_f])
    ax3.get_xaxis().set_visible(False)
    ax3.tick_params(labelsize=fontsize-3)
 
    if title != None:
        fig.tight_layout(rect=[0, 0.0, 1, 0.94])
        fig.suptitle(title,fontsize=fontsize)
        
    return fig



def plot_heatmaps(data, titles=None, N_cols=3, figsize=None, fontsizeT=13, fontsizeL=11, 
                  ylabel=None, xlabel=None, zlabel= None, cmap='jet',zmax=None, zmin=0,
                  xticks = None, yticks = None, zticks = None, N_rows = 'auto',
                  xticklabels = None, yticklabels=None, zticklabels=None,
                  suptitle=None,xticklab_rot=0):
    ''' Input:
        data:      3D array or list of 2D arrays
        titles:    array of titles, empty by default 
        N_cols:    number of columns, default 3
        figsize:   fixed figure size, will be determined automatically if None
        fontsizeT: fontsize for title
        fontsizeL: fontsize for labels
        xlab,ylab,clab: axis labels, empty by default, can be single string or list of strings 
        cmap:      name of a library cmap, or instance of mpc.LinearSegmentedColormap, or a list of either of these
    '''    
    
    data = np.array(data)
    N_plots = len(data)
    if N_rows == 'auto':
        N_rows  = int(np.ceil(1.*N_plots/N_cols) )
    
    N_plots2 = N_rows*N_cols
    # data2 = np.zeros([N_plots2,len(data[0]),len(data[0][0])])
    # data2[:N_plots] = data
    data2 = [[]]*N_plots2
    for i in range(N_plots):
        data2[i] = data[i]

    if figsize==None:
        figsize =[N_cols*4.8,N_rows*3.5]

        
    cmaps       = _repeat_if_needed(cmap, N_plots2, 1)    
    zmax        = _repeat_if_needed(zmax, N_plots2, 1)   
    zmin        = _repeat_if_needed(zmin, N_plots2, 1)   
    xticks      = _repeat_if_needed(xticks, N_plots2, 2)   
    yticks      = _repeat_if_needed(yticks, N_plots2, 2) 
    zticks      = _repeat_if_needed(zticks, N_plots2, 2)   
    xticklabels = _repeat_if_needed(xticklabels, N_plots2, 2)   
    yticklabels = _repeat_if_needed(yticklabels, N_plots2, 2)  
    zticklabels = _repeat_if_needed(zticklabels, N_plots2, 2)  

    fig,axes=plt.subplots(N_rows,N_cols,figsize=figsize)     
    plt.subplots_adjust(wspace=.2,hspace=.3)
    
    if type(xlabel) == str:
        xlabel = [xlabel] * N_plots2
    if type(ylabel) == str:
        ylabel = [ylabel] * N_plots2    
    if type(zlabel) == str:
        zlabel = [zlabel] * N_plots2  

    
    for i in range(N_plots2):    
        if (N_rows==1) or (N_cols ==1):
            ax = axes[i]
        else:
            ax = axes[i//N_cols,i%N_cols]            
        
       # ax.hold(True)
        ax.grid(False)
        if zmax[i] == None:
            zmax[i] = np.max(data2[i])
        p = ax.imshow(data2[i],origin='bottom',interpolation='none',cmap=cmaps[i],vmax=zmax[i],vmin=zmin[i])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.14)
        if np.all(zticks[i] != None):
            cb = plt.colorbar(p, cax=cax, ticks = zticks[i])  
            if zticklabels[i] != None:
                    cb.ax.set_yticklabels(zticklabels)
        else:
            zticks[i] = [zmax[i]*j for j in [0,1./4,1./2,3./4,1.]]
            cb = plt.colorbar(p, cax=cax, ticks = zticks[i])  
          
        try:            
            if np.all(titles!=None):
                ax.set_title(titles[i],fontsize=fontsizeT)
            if np.all(xlabel!=None):    
                ax.set_xlabel(xlabel[i], fontsize=fontsizeL)    
            if np.all(ylabel!=None):    
                ax.set_ylabel(ylabel[i], fontsize=fontsizeL)   
            if np.all(zlabel!=None):    
                cb.set_label(zlabel[i], fontsize=fontsizeL)  
            if np.all(xticks[i] !=None):
                ax.set_xticks(xticks[i])
            if np.all(yticks[i] !=None):
                ax.set_yticks(yticks[i])
            if np.all(xticklabels[i] !=None):
                ax.set_xticklabels(xticklabels[i],rotation=xticklab_rot)
            if np.all(yticklabels[i] !=None):
                ax.set_yticklabels(yticklabels[i])
            else:
                ax.set_yticks([])
        except:
             pass
    if suptitle != None:
         fig.suptitle(suptitle,fontsize=18)
   
def semi_log_plot(figsize, data, freqs, 
                  xlim=[1,100], ylabel='variable', ylim=None, CI=None,
                  legend=None, legend_pos='best', cmap='gist_rainbow',
                  ncols=3, xticks=None, bgcolor=None, title=None,yticks='none',
                  fontsize=11, markersize=0, show=True, outfile='none',
                  sig_id="none", sig_fac=1.05, sig_style='*', plot_zero=True):   
    '''
    plots data over log-spaced frequencies; 
    with or without confidence intervals (or standard deviation)
    INPUT:
      figsize:       figure size [width, height]
      data:          2D or 3D array/list (without or with CI/SD, resp.)
                     either [groups x freqs]
                     or [groups x (mean, lower bound, upper bound) x freqs]
      freqs:         1D or 2D array/list of frequencies (float)
                     use 2D if different frequencies for different groups                         
      xlim:          [xmin,xmax]
      ylim:          [ymin,ymax]
      ylabel:        label for the y axis    
      CI:            None if no CI/SDs, else alpha value (0 to 1) for CI/SD areas
      legend:        array of strings, 1 for each group
      legend_pos:    position of legend ('uc','br' or 'ur'); no legend if None
      cmap:          either name of a standard colormap 
                     or an instance of matplotlib.colors.LinearSegmentedColormap
      ncols:         number of columns in the plot legend
      xticks:        custom values for xticks. if None, standard value are used
      bgcolor:       background color
      fontsize:      fontsize
      markersize:    size of data point marker, default = 0
      show:          if False, the figure is not shown in console/window
      outfile:       if not None, figure will be exported to this file        
      sig_id:        indices where significance is to be indicated
                      can be 2D as [sig,sig_c], then sig in black, sig_c in red
      sig_fac:       controls the height at which significance indicators shown
      sig_style:     if '*' or '-' or '-*': indicated by stars above ploted lines
                     if 's' or 'o': indicated as markers on the plotted lines
      plot_zero:     whether to draw the x-axis    
    OUTPUT:
      fig: instance of matplotlib figure        
                   
    '''
    depth = lambda L: isinstance(L, list) and max(map(depth, L))+1
    fig,ax=plt.subplots(figsize=figsize) 
    if sig_id != 'none':
        if _depth(sig_id)==1:
            sig_id[0]=sig_id[0]-sig_id[1]
        sig_id=np.array(sig_id)
        sig_id[np.where(sig_id==0)]=np.nan    
    if type(cmap) is mpc.LinearSegmentedColormap:
        colorspace = [cmap(i) for i in np.linspace(0, 1, len(data))]            # get colorspace from colormap
    else:
        colorspace = [plt.get_cmap(cmap)(i) for i in np.linspace(0, 1, len(data))]   
    if CI != None:
        colorspace_CI = np.array(colorspace)*np.array([1,1,1,CI])                              # colors for confidence intervals
    ax.set_prop_cycle(color=colorspace)                                                          # set different colors for different plots
    for i in range(len(data)):                                                              # for each plot i
        if depth(freqs)==2:    
            freqs2=freqs[i]
        else:
            freqs2=freqs
        if CI != None: 
            N_F = len(data[i][0])
            ax.plot(freqs2[:N_F],data[i][0],'o-',markersize=markersize)                                     # if CI, data for each plot i comes as [mean,CI_low, CI_high]
            ax.fill_between(freqs2,data[i][1],data[i][2],color=colorspace_CI[i])             # fill between CI_low and CI_high
        else:
            N_F = len(data[i])
            ax.plot(freqs2[:N_F],data[i],'o-',markersize=markersize)   
            
    if plot_zero:
        ax.plot(freqs2,np.zeros(N_F),'k')            
    if xticks==None:
        xticks=[1,2,3,5,10,20,30,50,100,200,300]    
    if yticks!='none':
        ax.set_yticks(yticks)
    ax.set_xscale('log')
    ax.set_xticks(xticks)
    if bgcolor != None:
#        ax.set_axis_bgcolor(bgcolor)
        ax.set_facecolor(bgcolor)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(xlim) 
    ax.axis('on')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel('Frequency [Hz]', fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    if title !=None:
        plt.title(title,fontsize=fontsize+2)
    if type(sig_id)!=str:
        if _depth(sig_id)==1:
            plt.plot(freqs[:len(sig_id)],sig_id*np.nanmax(data)*sig_fac,
                       sig_style,color='k')
        else:
            sig_id = np.array(sig_id)
            plt.plot(freqs[:len(sig_id[0])],(sig_id[0])*np.nanmax(data)*sig_fac,
                       sig_style,color='k')
            plt.plot(freqs[:len(sig_id[0])],(sig_id[1])*np.nanmax(data)*sig_fac,
                       sig_style,color='r')
    if np.any(ylim!=None):
        ax.set_ylim(ylim) 
        
    loc_dict = {'uc': 'upper center', 'ur': 'upper right', 'ul':'upper left', 
                'bc': 'lower center', 'br': 'lower right', 'bl':'lower left',
                'best': 'best'}  
    if np.any(legend!=None):
        plt.legend(legend, loc=loc_dict.get(legend_pos), ncol=ncols, fontsize=fontsize-2) 
    if outfile!='none':    
        plt.savefig(outfile) 
    if show:
        plt.show()
    plt.close()
    return fig    
        
        
def semi_log_plot2(figsize,data,freqs,xlim=[1,100],ylabel='variable',legend=None,outfile=None,
                  legend_pos='best',ylim=None,show=True,cmap='gist_rainbow',
                  ncols=3,CI=None,xticks=None,bgcolor=None,
                  fontsize=11, markersize=0,title=None,
                  sig_id="none",sig_fac=1.05,sig_style='*',plot_zero=True):   
    '''
    plots data over log-spaced frequencies with sig. indices on the plot lines
    
    figsize:       figure size [width, height]
    data:          2D  array/list [groups x frequencies]
    freqs:         1D or 2D array/list of frequencies (float)
                   use 2D if different frequencies for different groups                         
    xlim:          [xmin,xmax]
    ylabel:        label for the y axis
    legend:        array of strings, 1 for each group
    outfile:       if not None, figure will be exported to this file
    legend_pos:    position of legend ('uc','br' or 'ur'); no legend if None
    ylim:          [ymin,ymax]
    show:          if False, the figure is not shown in console/window
    cmap:          either name of a standard colormap 
                   or an instance of matplotlib.colors.LinearSegmentedColormap
    ncols:         number of columns in the plot legend
    xticks:        custom values for xticks. if None, standard value are used
    bgcolor:       background color
    fontsize:      fontsize
    markersize:    size of data point marker, default = 0
    sig_style:     None or 1D or 2D e.g. 's' or 'o', or ['s','o']
    sig_id:        None or array of indices where significance is to be indicated
                    can be 2D as [group x freq]
                        or 3D as [2x group x freq] with 1st = [sig,sig_c]
    sig_fac:       controls the height at which significance indicators shown
    '''
    fig,ax=plt.subplots(figsize=figsize) 
    N_groups = len(data)
    if type(cmap) is mpc.LinearSegmentedColormap:
        colorspace = [cmap(i) for i in np.linspace(0, 1, len(data))]            # get colorspace from colormap
    else:
        colorspace = [plt.get_cmap(cmap)(i) for i in np.linspace(0, 1, len(data))]
    if sig_style!=None:
        if _depth(sig_style)==0:
            sig_data = data*sig_id
            sig_data[np.where(sig_data==0)]=np.nan
            sig_style1 = sig_style
            colorspace = colorspace+colorspace
        else:
            sig_data  = data*sig_id[0]
            sig_dataC = data*sig_id[1]
            sig_data[np.where(sig_data==0)]=np.nan
            sig_dataC[np.where(sig_dataC==0)]=np.nan
            sig_style1 = sig_style[0]
            sig_style2 = sig_style[1]        
            colorspace = colorspace+colorspace+colorspace
        
    ax.set_prop_cycle(color=colorspace)                                                          # set different colors for different plots
    for i in range(N_groups):                                                              # for each plot i
        if _depth(freqs)==2:    
            freqs2=freqs[i]
        else:
            freqs2=freqs
        N_F = len(data[i])
        ax.plot(freqs2[:N_F],data[i],'-',markersize=markersize)     
        
    if (sig_style != None) :
        for i in range(N_groups):                                                              # for each plot i
            if _depth(freqs)==2:    
                freqs2=freqs[i]
            else:
                freqs2=freqs
            N_F = len(data[i]) 
            ax.plot(freqs2[:N_F],sig_data[i],linestyle='',
                    marker=sig_style1,markersize=markersize) 
    if (sig_style != None) & (_depth(sig_style)==1):
        for i in range(N_groups):                                                              # for each plot i
            if _depth(freqs)==2:    
                freqs2=freqs[i]
            else:
                freqs2=freqs
            N_F = len(data[i])   
            ax.plot(freqs2[:N_F],sig_dataC[i],linestyle='',
                    marker=sig_style2,markersize=markersize) 
    if plot_zero:
        ax.plot(freqs2,np.zeros(N_F),'k')
         
    if xticks==None:
        xticks=[1,2,3,5,10,20,30,50,100,200,300]
    ax.set_xscale('log')
    ax.set_xticks(xticks)
    if bgcolor != None:
#        ax.set_axis_bgcolor(bgcolor)
        ax.set_facecolor(bgcolor)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(xlim) 
    ax.axis('on')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel('Frequency [Hz]', fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    if title !=None:
        plt.title(title,fontsize=fontsize+2)
    
    if ylim!=None:
        ax.set_ylim(ylim) 
        
    loc_dict = {'uc': 'upper center', 'ur': 'upper right', 'ul':'upper left', 
                'bc': 'lower center', 'br': 'lower right', 'bl':'lower left',
                'best': 'best'}  
    if legend!=None:
        plt.legend(legend, loc=loc_dict.get(legend_pos), ncol=ncols, fontsize=fontsize) 
    if outfile!=None:    
        plt.savefig(outfile) 
    if show:
        plt.show()
    plt.close()
    return fig            

    
    
    

def semi_log_plot_multi(figsize,rows,cols,dataL,freqs,xlimA,ylabA,titlesA,
                        cmapA,legendA=None,outfile=None,legend_posA=None,
                        ylimA=None,show=False,ncols=3,CI=None,
                        xlabA=None,Ryt=None,xticks='auto',fontsize=8,
                        markersize=0,sig_idA=None,sig_fac=1,sig_style='*'): 
    '''
    multiple subplots of data over log-spaced frequencies
    with or without confidence intervals/standard deviation 
    
    figsize:       figure size [width, height]
    rows:          number of rows for subplots
    cols:          number of columns for subplots
    dataL:         3D or 4D array/list (without or with CI/SD resp.)
                   1st dim: datasets, 1 per subplot
                   2nd dim: groups within subplot 
                   optional dim: [mean, lower bound, upper bound] for CI or SD
                   last dim: frequencies   
                   The numbers of groups and frequencies can vary between 
                   subplots, if your dataL object is a list on the 1st dim.
    freqs:         1D, 2D or 3D array/list of frequencies (float) 
                   2D if every group uses different frequencies
                   3D if every dataset and every group uses different freqs 
                   Dimensions must match the data!
    xlimA:         2D array of [xmin,xmax] for each subplot
    ylabA:         2D array of labels for the y axis in each subplot
    titlesA:       2D array of subplots titles    
    cmapA:         array of colormaps, either names of standard colormaps 
                   or instances of matplotlib.colors.LinearSegmentedColormap
    legendA:       2D array of legends (strings); or None for no legends 
    outfile:       if not None, figure will be exported to this file
    legend_posA:   position of the legend ('uc' or 'ur') in each subplot; 
                   or None for no legends
    ylimA:         2D array of [ymin,ymax] for each subplot; or None for auto
    show:          if False, the figure is not shown in console/window

    ncols:         number of columns in the plot legend
    CI:            None if no CI/SDs, else alpha value (0 to 1) for CI/SD areas 
    xticks:        custom values for xticks. If auto, standard values are used
    xlabA:         array of booleans, whether to show the x label; 
                   or None for all True
    Ryt:           if not None, reduces the number of y ticks
    fontsize:      fontsize in plot
    markersize:    size of markers in plot
    '''
    
    depth = lambda L: isinstance(L, list) and max(map(depth, L))+1
    fig,axes=plt.subplots(rows,cols,figsize=figsize)
    if CI == None:
        CI = [None for i in range(len(dataL))]  
    if ylimA==None:
        ylimA = [False for i in range(len(dataL))]
    if xlabA==None:
        xlabA = [True for i in range(len(dataL))]
    if legend_posA==None:
        legend_posA = [None for i in range(len(dataL))]  
    if sig_idA==None:
        sig_idA = ['none' for i in range(len(dataL))]  

    for d,data in enumerate(dataL):         # each dataset in one subplot 
        if (rows==1) or (cols ==1):
            ax = axes[d]
        else:
            ax = axes[d//cols,d%cols]
#        ax.hold(True)
        ax.set_title(titlesA[d],fontsize=fontsize)
        
        if type(cmapA[d]) is mpc.LinearSegmentedColormap:
            colorspace = [cmapA[d](i) for i in np.linspace(0, 1, len(data))]
        else:
            colorspace = [plt.get_cmap(cmapA[d])(i) for i in np.linspace(0, 1, len(data))]
        if CI[d]!= None:
            colorspace_CI = np.array(colorspace)*np.array([1,1,1,CI[d]])
        #ax.set_color_cycle(colorspace)
        ax.set_prop_cycle(color=colorspace) 
        for i in range(len(data)):
            if depth(freqs)==3:
                freqs2=freqs[d][i]
            elif depth(freqs)==2:    
                freqs2=freqs[i]
            else:
                freqs2=freqs
                
            if CI[d]!=None:
                fr = freqs2[:len(data[i][0])]
                ax.plot(fr,data[i][0],'o-',markersize=markersize)
                ax.fill_between(fr,data[i][1],data[i][2],color=colorspace_CI[i])    
            else:
                fr = freqs2[:len(data[i])]
                ax.plot(fr,data[i],'o-',markersize=markersize,color=colorspace[i])
            
            sig_id = sig_idA[d]
            if type(sig_id)!=str:
                if _depth(sig_id)==1:
                    ax.plot(freqs[:len(sig_id)],sig_id*np.nanmax(data)*sig_fac,
                               sig_style,color='k')
                else:
                    sig_id = np.array(sig_id)
                    ax.plot(freqs[:len(sig_id[0])],(sig_id[0])*np.nanmax(data)*sig_fac,
                               sig_style,color='k')
                    ax.plot(freqs[:len(sig_id[0])],(sig_id[1])*np.nanmax(data)*sig_fac,
                               sig_style,color='r')
        if Ryt != None:
            if Ryt[d] ==1:
                for label in ax.get_yticklabels()[::2]:
                    label.set_visible(False)  
                 
        ax.set_xscale('log')
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter()) 
        if xticks=='auto':
            xticks=[1,2,3,5,10,20,30,50,100,200,300]
        ax.set_xticks(xticks)
        xticklabels = [str(i) for i in xticks]
        ax.set_xticklabels(xticklabels,fontsize=fontsize-2)
        if xlabA[d]==True:
            ax.set_xlabel('Frequency [Hz]',fontsize=fontsize)       
        ax.set_xlim(xlimA[d]) 
        ax.set_ylabel(ylabA[d],fontsize=fontsize)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if np.any(ylimA[d]!=None):
            ax.set_ylim(ylimA[d]) 
            
        loc_dict = {'uc': 'upper center', 'ur': 'upper right', 
                'br': 'lower right', 'best': 'best'}  
        if legend_posA[d] !=None:
            if legendA[d] != None:
                ax.legend(legendA[d],loc=loc_dict.get(legend_posA[d]), 
                          bbox_to_anchor=(0.5, 1.05), ncol=ncols)
        if legend_posA[d]=='ur':
            if legendA[d] != None:
                ax.legend(legendA[d], loc='upper right', ncol=ncols, frameon=0,fontsize=fontsize)  
        if legend_posA[d]=='ul':
            if legendA[d] != None:
                ax.legend(legendA[d],loc='upper left', ncol=ncols, frameon=0,fontsize=fontsize)
    plt.tight_layout()   
    if outfile!=None:    
        plt.savefig(outfile)
    if show:
        plt.show()
  
    plt.clf()
    


def plot_histogram(data, N_bins=20, width=0.7):
    import matplotlib.pyplot as plt
    import numpy as np
    
    hist, bins = np.histogram(data, bins=N_bins)
    width = width * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) // 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()
    
    
    
    
    
    



def simple_CF_plot(data,figsize,xlabel,ylabel,xtix,ytix,xticklabels,yticklabels,zmax=1,ztix=[0,0.2,0.4,0.6,0.8,1],
                   outfile=None,cmap='std',zmin=None,fontsize=10,zlabel=None,title=None):            
    
    #data in shape freq x ratio
    # example of use: 
    # LF_indices = [0,4,8,12,16,20,24,28,32,36,40]
    # LF_str     = ['1.1', '2.2', '3.7', '5.9', '9.0', '13.1', '19.7', '28.7', '42.5', '65.3', '95.6']
    # plots.simple_CF_plot(data,figsize,'ratio','LF',np.arange(0.5,5.6,1),LF_indices,ratios,LF_str,zmax=zmax,ztix=ztix,outfile=None)             


    #eps  = np.spacing(0.0)                                            # an epsilon
    if zmin==None:
        vmin = 1e-20
    else:
        vmin = zmin
    if cmap == 'std':
        CM = plt.cm.YlOrRd 
        CM.set_under('None')
    else:
        CM = cmap
    
    fig=plt.figure(figsize=figsize, facecolor='w', edgecolor='k')
    ax=fig.add_axes([0.2,0.2,.7,.7])
    mesh=ax.pcolormesh(data,vmin=vmin,vmax=zmax,cmap=CM)    
    ax.set_xticks(xtix)
    ax.set_xticklabels(xticklabels,rotation=45,fontsize=fontsize)
    ax.set_xlim([0,len(data[0])])      
    ax.set_yticks(ytix)
    ax.set_yticklabels(yticklabels,fontsize=fontsize)
    ax.set_ylim([0,len(data)])
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.tick_params(axis='both',which='both',length=0)
    ax.set_ylabel(ylabel,fontsize=fontsize)    
    cbar = fig.colorbar(mesh, ticks=ztix)
    cbar.ax.tick_params(axis='y', direction='out',labelsize=fontsize )
    
    if title != None:
        fig.suptitle(title)
    
    
    if zlabel !=None:        
        cbar.set_label(zlabel)
    if outfile!=None:
        fig.savefig(outfile)




def _repeat_if_needed(param,N_plots,depth):   
    if _depth(param)==0:
        if param==None and depth>0:
            depth=1     
    if _depth(param) >= depth:
        param = param
    else:
        param = [param for n in range(N_plots)]
        param = _repeat_if_needed(param,N_plots,depth)
    return param
    
    
def _depth(obj,c=0):
     try:
         if type(obj) != str:
             obj1 = obj[0]
             c = _depth(obj1,c+1)
         else:
             return c
     except:
         return c
     return c
 
    
 