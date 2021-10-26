# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:59:28 2020
@author: Felix Siebenhühner
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_functions as plots
import enum


class Parc:
    def __init__(self,name,dir1='L:\\nttk_palva\\Utilities\\parcellations\\parc_info\\'
                 ,suffix='',split=False):
        self.name = name        
        all_names = np.genfromtxt(dir1 + name + suffix + '.csv',delimiter=';',dtype='str')
        if name == 'parc2009':
            self.N             = 148
            self.names         = all_names[:,0]
            self.abbr          = all_names[:,1]
            self.networks      = (all_names[:,2]).astype('int')
            self.netw_names    = ['Control','Default','Dorsal Attention','Limbic','Ventral Attention','Somatomotor','Visual']
            self.netw_abbr     = ['Con','DMN','DAN','Lim','VAN','SMN','Vis']
            self.netw_masks    = get_network_masks(self.networks,7,self.N)
            self.netw_indices  = [np.where(self.networks==i)[0] for i in range(7)]
            self.NN            = np.sum(self.netw_masks,(2,3))
            try:
                self.lobe     = all_names[:,3]
            except:
                pass

        else:
            self.N = int(name[-3:])
            self.names = all_names[:,0]
            self.abbr  = all_names[:,1]
            
            if '2018yeo7' in name:        
                self.netw_names   = ['Control','Default','Dorsal Attention',
                                      'Limbic','Ventral Attention','Somatomotor','Visual']
                self.netw_abbr2   = ['Con','DMN','DAN','Lim','VAN','SMN','Vis']
                self.netw_abbr    = ['CT' ,'DM' ,'DA' ,'Lim','VA' ,'SM' ,'Vis']
                self.networks     = get_networks(self.abbr,self.netw_abbr)
                self.N_netw       = 7
                self.netw         = 'yeo7'
                self.netw_indices = [np.where(self.networks==i)[0] for i in range(7)]
                self.netw_masks   = get_network_masks(self.networks,7,self.N)
                self.NN           = np.sum(self.netw_masks,(2,3))
                
                #self.lobe         = np.concatenate((all_names[::2,4],all_names[1::2,4]))
                
            elif '2018yeo17' in name and not split:
                self.netw_names   = ['ContA','ContB','ContC','DefaultA','DefaultB','DefaultC','DorsAttnA',
                                      'DorsAttnB','LimbicA_TempPole','LimbicB_OFC','SalVentAttnA','SalVentAttnB',
                                      'SomMotA','SomMotB','TempPar','VisCent','VisPeri']
                self.netw_abbr    = ['CT-A','CT-B','CT-C','DM-A','DM-B','DM-C','DA-A',
                                      'DA-B','LimA','LimB','VA-A','VA-B',
                                      'SM-A','SM-B','TPar','VisA','VisB']
                self.networks     = get_networks(self.names,self.netw_names)
                self.netw         = 'yeo17'
                self.netw_indices = [np.where(self.networks==i)[0] for i in range(17)]
                self.N_netw       = 17
                self.netw_masks   = get_network_masks(self.networks,17,self.N)  
                self.NN           = np.sum(self.netw_masks,(2,3))

                try:
                    self.lobe     = all_names[:,2]
                except:
                    pass
                
            elif '2018yeo17' in name and split:
                self.netw_names   = ['ContA','ContB','ContC','DefaultA','DefaultB',
                                     'DefaultC','DorsAttnA','DorsAttnB','LimbicA_TempPole',
                                     'LimbicB_OFC','SalVentAttnA','SalVentAttnB',
                                     'SomMotA','SomMotB','TempPar','VisCent','VisPeri']

                
                self.netw_abbr    = ['CT-A','CT-B','CT-C','DM-A','DM-B','DM-C','DA-A',
                                      'DA-B','LimA','LimB','VA-A','VA-B',
                                      'SM-A','SM-B','TPar','VisA','VisB']
                self.networks     = get_networks_HS(self.names,self.netw_names)                
                
                self.netw_names   = [n + '_L' for n in self.netw_names] + \
                                    [n + '_R' for n in self.netw_names]
                self.netw_abbr    = [n + '_L' for n in self.netw_abbr] + \
                                    [n + '_R' for n in self.netw_abbr]
                self.netw         = 'yeo34'
                self.netw_indices = [np.where(self.networks==i)[0] for i in range(34)]
                self.N_netw       = 34
                self.netw_masks   = get_network_masks(self.networks,34,self.N)  
                self.NN           = np.sum(self.netw_masks,(2,3))
                self.name         = self.name + '_s' 

                try:
                    self.lobe     = all_names[:,2]
                except:
                    pass

             
    def __repr__(self):
        return ('\'' + self.name + '\'')
    
    

def get_networks(parcel_names,system_names):            
    net_ind = np.zeros(len(parcel_names))
    for s, system in enumerate(system_names):
        for p, parcel in enumerate(parcel_names):
            if system in parcel:
                net_ind[p] = s
    return net_ind.astype('int')

def get_networks_HS(parcel_names,system_names):            
    net_ind = np.zeros(len(parcel_names))
    for s, system in enumerate(system_names):
        for p, parcel in enumerate(parcel_names):
            if system in parcel and 'LH' in parcel:
                net_ind[p] = s
            elif system in parcel and 'RH' in parcel:
                net_ind[p] = s + len(system_names)
    return net_ind.astype('int')

def get_network_masks(network_indices,N_netw,N_parc):
    network_masks = np.zeros([N_netw,N_netw,N_parc,N_parc])
    for i in range(N_netw):
        for j in range(N_netw):
            network_masks[i,j] = np.matmul(np.transpose([network_indices==i]),[network_indices==j]).astype('int')
    return network_masks    


def get_fidelity_and_cpp(project_dirs,set_sel,parc,suffix):
    
    N_sets = len(set_sel)
    fid = np.zeros([N_sets,parc.N])
    cpp = np.zeros([N_sets,parc.N,parc.N])

    
    for s in range(N_sets):
        pd      = project_dirs[s]
        set1    = set_sel[s]
        subject = set1[:5]
        dir1    = pd + subject + '\\Fidelity\\' 
        file1   = dir1 + 'Fidelity_' + set1 + suffix + '_' + parc.name + '.csv'
        file2   = dir1 + 'CP-PLV_' + set1 + suffix + '_' + parc.name + '.csv'
        fid[s]  = np.genfromtxt(file1,delimiter=';') 
        cpp[s]  = np.genfromtxt(file2,delimiter=';') 
    ## get fid*fid product
    fidsq = np.zeros([len(fid),parc.N,parc.N])
    for s in range(len(fid)):
        fidsq[s] = np.outer(fid[s],fid[s])
    return fid, cpp, fidsq
   
def plot_fid_and_cpp(fid,cpp,fidsq):
    

    
    fid_mean     = np.mean(np.array(fid),0)   
    fid_mean_ps  = np.mean(np.array(fid),1)  
    cpp_mean     = np.mean(np.array(cpp),0)
    cpp_mean_ps  = np.mean(np.array(cpp),(1,2))    
    
    plots.plot_lines(fid_mean,ylabel='Fidelity',xlabel='parcel',
                     title='Mean fidelity per parcel',fontsize=14)
    plt.show()
    plots.plot_lines(fid_mean_ps,ylabel='Fidelity',xlabel='set',
                     title='Mean parcel fidelity per set',fontsize=14)
    plt.show()
    plots.plot_lines(cpp_mean_ps,ylabel='CP-PLV',xlabel='set',
                     title='Cross-parcel PLV',fontsize=14)
    plt.show()
    plots.plot_heatmap(np.mean(fidsq,0),xlabel='parcel',ylabel='parcel',
                       zlabel='Fidelity product',title='Fidelity product',
                       figsize=[9,7],cmap='viridis',fontsize=16)        
    plt.show()
    plots.plot_heatmap(cpp_mean,xlabel='parcel',ylabel='parcel',zlabel='cp-PLV',
                       figsize=[9,7],cmap='viridis',fontsize=16,
                       title='Cross-parcel PLV')
    plt.show()
    
    plt.hist(np.reshape(fid,np.prod(np.shape(fid))),bins=60)
    plt.title('Parcel fidelity')
    plt.show()
    plt.hist(np.reshape(cpp,np.prod(np.shape(cpp))),bins=60)  
    plt.title('Cross-parcel PLV')
    plt.show()
    
def make_mask(fid,cpp,fid_threshold,cpp_threshold,parc):  

    fid_mean   = np.mean(np.array(fid),0)   
    cpp_mean   = np.mean(np.array(cpp),0)
    fid_mask   = np.outer((fid_mean>fid_threshold),(fid_mean>fid_threshold))   
    cpp_mask   = cpp_mean < cpp_threshold                        
    mask       = 1* fid_mask * cpp_mask
    np.fill_diagonal(mask,0)
    edges_retained = np.sum(mask)/float(parc.N**2-parc.N)
    print ("Edges retained: " + '{:.2f}'.format(100*edges_retained) + '%')
    print ("Edges rejected: " + '{:.2f}'.format(100-100*edges_retained) + '%')
    plots.plot_heatmap(mask,xlabel='parcel',ylabel='parcel',
                       figsize=[7,7],cmap='viridis',fontsize=16,cbar=None)
 
    return mask

def save_mask(mask,base_dir,parc,fid_threshold,cpp_threshold,N):
    fileout_DEM = base_dir + '_settings\\masks\\' + parc.name + ' fid' \
        + str(fid_threshold) + ', cp' + str(cpp_threshold) \
                    + ', N=' +str(N) + '.csv' 
    np.savetxt(fileout_DEM,mask,delimiter=';',fmt='%i')



def get_mean_netw(values,parc,axis=1):
    ''' get mean over an axis for each network
    INPUT: 
        values: 2D array [freqs x parcels]
        parc:   a Parcellation object
    '''
    
    a = np.array([np.nanmean(values[:,parc.netw_indices[n]],axis) for n in range(parc.N_netw)])   
    
    return a

def get_mean_NN(values,parc):
    ''' get mean K for each network pair
    INPUT: 
        values: 3D array [freqs x parcels x parcels]
        parc:   a Parcellation object
    '''
    
    N_freq = len(values)
    K = np.zeros([N_freq,parc.N_netw,parc.N_netw])
    
    for f in range(N_freq):
        for n1 in range(parc.N_netw):
            for n2 in range(parc.N_netw):
                K[f,n1,n2] = np.nansum(values[f] * parc.netw_masks[n1,n2])/parc.NN[n1,n2]
    
    return K

def get_K_NN(values,parc):
    ''' get mean K for each network pair
    INPUT: 
        values: 3D array [freqs x parcels x parcels]
        parc:   a Parcellation object
    '''
    
    N_freq = len(values)
    K = np.zeros([N_freq,parc.N_netw,parc.N_netw])
    
    for f in range(N_freq):
        for n1 in range(parc.N_netw):
            for n2 in range(parc.N_netw):
                K[f,n1,n2] = np.nansum((values[f] != 0) * parc.netw_masks[n1,n2])/parc.NN[n1,n2]
    
    return K




def get_reordering(parcels_source,parcels_target):
    '''
    INPUT:  1D lists or arrays of parcels
    OUTPUT: 2 x 1D array of indices mapping old to new, and new to old, resp
    '''
       
    N_parc     = len(parcels_source)
    old_to_new = np.full(N_parc,-1,'int')
    new_to_old = np.full(N_parc,-1,'int')
    for i in range(N_parc):
        found = 0
        for j in range(N_parc):            
            if parcels_source[i] == parcels_target[j]:
                old_to_new[j] = i
                new_to_old[i] = j
                found = 1
        if found == 0:
            print('Not found: ' + parcels_source[i])
                
    return old_to_new, new_to_old
















    
    