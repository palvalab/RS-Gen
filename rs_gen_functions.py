# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:50:08 2021
@author: Felix Siebenhühner
"""
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.sandbox.stats.multicomp as multicomp
import scipy.stats as st

#%% plot abs synch values

def plot_comt_values(N_bands,mean_comt_pf,f_band_str,comt_groups,network_names,vmaxA):   
    fig,ax = plt.subplots(N_bands,3,figsize=[9,10])
    for i in range(N_bands):
        for j in range(3):
            vmin = 0
            vmax = vmaxA[i]
            vdif = vmax-vmin
            im=ax[i][j].imshow(mean_comt_pf[i][j],interpolation='none',vmin=vmin,
                               vmax=vmax,origin='bottom',cmap='jet')
            ax[i][j].grid(linestyle='')
            if i==3:
                ax[i][j].set_xticks(np.arange(0,6.1,1))        
                ax[i][j].set_xticklabels(network_names,rotation=45);
            else:
                ax[i][j].set_xticks([])
            if j==0:
                ax[i][j].set_yticks(np.arange(0,6.1,1))
                ax[i][j].set_yticklabels(network_names);
            else:
                ax[i][j].set_yticks([])
    
            if j==2:
                cticks=[vmin,vmin+(vdif/3),vmin+(2*vdif/3),vmax]
                cb =  fig.colorbar(im, ax=ax[i,:], ticks=cticks,
                                   location = 'right',shrink=1)
    
            # cb.locator = mpl.ticker.MaxNLocator(nbins=3)
            # cb.update_ticks()
    
            if i==0:
                ax[i][j].set_title(comt_groups[j])
    
        ax[i][0].set_ylabel(f_band_str[i])
    return fig
         
        
def plot_bdnf_values(N_bands,mean_bdnf_pf,f_band_str,bdnf_groups,network_names,vmaxA):    
    fig,ax = plt.subplots(N_bands,2,figsize=[6,10])
    for i in range(N_bands):
        for j in range(2):
            vmin = 0
            vmax = vmaxA[i]
            vdif = vmax-vmin
            im=ax[i][j].imshow(mean_bdnf_pf[i][j],interpolation='none',
                               vmin=vmin,vmax=vmax,origin='bottom',cmap='jet')
            ax[i][j].grid(linestyle='')
            if i==3:
                ax[i][j].set_xticks(np.arange(0,6.1,1))        
                ax[i][j].set_xticklabels(network_names,rotation=45);
            else:
                ax[i][j].set_xticks([])
            if j==0:
                ax[i][j].set_yticks(np.arange(0,6.1,1))
                ax[i][j].set_yticklabels(network_names);
            else:
                ax[i][j].set_yticks([])    
            if j==1:
                cticks=[vmin,vmin+(vdif/3),vmin+(2*vdif/3),vmax]
                cb =  fig.colorbar(im, ax=ax[i,:], ticks=cticks,
                                   location = 'right',shrink=1)
            if i==0:
                ax[i][j].set_title(bdnf_groups[j])
        ax[i][0].set_ylabel(f_band_str[i])  
    return fig


#%% plot synch diff      
        
def plot_comt_diff(N_bands,mean_comt_pf,f_band_str,comt_groups,network_names,vmaxA):
        
    fig,ax = plt.subplots(N_bands,3,figsize=[12,12])
    for i in range(N_bands):
        for j in range(3):
            # vmax1 = abs(np.max(mean_comt_pf[i][j]-np.mean(mean_comt_pf[i],0)))
            # vmax1 = 0.012
            vmax = vmaxA[i]
            vmin = -0.00
            vdif = vmax-vmin
            im=ax[i][j].imshow(mean_comt_pf[i][j]-np.mean(mean_comt_pf[i],0),
                               vmin=-vmax,vmax=vmax,interpolation='none',
                               origin='bottom',cmap='jet')
            ax[i][j].grid(linestyle='')
            ax[i][j].set_xticks(np.arange(0,6.1,1))
            ax[i][j].set_yticks(np.arange(0,6.1,1))
            ax[i][j].set_xticklabels(network_names);
            ax[i][j].set_yticklabels(network_names);
            if j==2:
                fig.colorbar(im, ax=ax[i][j])
            cticks=[vmin,vmin+(vdif/3),vmin+(2*vdif/3),vmax]
            # cb =  fig.colorbar(im, ax=ax[i,:], ticks=cticks,
            #                    location = 'right',shrink=1)
            if i==0:
                ax[i][j].set_title(comt_groups[j]+' - mean')
        ax[i][0].set_ylabel(f_band_str[i]) 
    return fig
            
def plot_bdnf_diff(N_bands,mean_bdnf_pf,f_band_str,bdnf_groups,network_names,vmaxA):
   
    fig,ax = plt.subplots(N_bands,2,figsize=[10,12])    
    for i in range(N_bands):
        for j in range(1):
           # vmax1 = abs(np.max(mean_bdnf_pf[i][j]-np.mean(mean_bdnf_pf[i],0)))
            vmax = vmaxA[i]
            vmin = -0.00
            vdif = vmax-vmin
            plot_data = mean_bdnf_pf[i][j]-(mean_bdnf_pf[i][j+1])
            im=ax[i][j].imshow(1*plot_data,vmin=vmin,vmax=vmax,interpolation='none',
                               origin='bottom',cmap='jet')
            ax[i][j].grid(linestyle='')
            ax[i][j].set_xticks(np.arange(0,6.1,1))
            ax[i][j].set_yticks(np.arange(0,6.1,1))
            ax[i][j].set_xticklabels(network_names);
            ax[i][j].set_yticklabels(network_names);
            cticks=[vmin,vmin+(vdif/3),vmin+(2*vdif/3),vmax]
            fig.colorbar(im, ax=ax[i][j],ticks=cticks)
            if i==0:
                ax[i][j].set_title(bdnf_groups[j])
        ax[i][0].set_ylabel(f_band_str[i])   
    return fig              
        
        
#%% tests for synch   
        
def synch_test_comt(data1a,N_bands,f_bands,comt_idx,network_names,f_band_str,corr_method='fdr_bh'):
    h_krusk_comt_pn   = np.zeros([N_bands,2,7,7])
    p_krusk_comt_pn   = np.zeros([N_bands,2,7,7])
    
    for fb in range(N_bands):
        for n1 in range(7):
            for n2 in range(7):
                a = np.array([[ np.mean([data1a[f][s][n1][n2] for f in f_bands[fb]]) for s in comt_idx[k]] for k in range(3)])
                try:
                    h,p = st.kruskal(a[0],a[1],a[2])
                except:
                    h=0
                    p=1
                h_krusk_comt_pn[fb,0,n1,n2] = h
                p_krusk_comt_pn[fb,0,n1,n2] = p
        p1  = p_krusk_comt_pn[fb][0]
        ixu = np.triu_indices(7)
        p2  = p1[ixu]
        corrv = (multicomp.multipletests(p2,method=corr_method)[1])    
        for iu in range(28):  
            p_krusk_comt_pn[fb,1,ixu[0][iu],ixu[1][iu]] = corrv[iu]
            p_krusk_comt_pn[fb,1,ixu[1][iu],ixu[0][iu]] = corrv[iu]    
    
    # plot sign COMT differences  
    fig,ax = plt.subplots(N_bands,2,figsize=[10,12])
    for fb in range(N_bands):
        for j in range(2):
            im=ax[fb][j].imshow( (p_krusk_comt_pn[fb][j]<0.05),vmin=0,vmax=1,interpolation='none',origin='bottom',cmap='Reds')
            ax[fb][j].grid(linestyle='')
            ax[fb][j].set_xticks(np.arange(0,6.1,1))
            ax[fb][j].set_yticks(np.arange(0,6.1,1))
            ax[fb][j].set_xticklabels(network_names);
            ax[fb][j].set_yticklabels(network_names);
            fig.colorbar(im, ax=ax[fb][j])
        ax[fb][j].set_ylabel(f_band_str[fb])  

    return fig           
                  
def synch_test_bdnf(data1a,N_bands,f_bands,bdnf_idx,network_names,f_band_str,corr_method='fdr_bh'):
    
    # compute for BDNF
    h_krusk_bdnf_pn  = np.zeros([N_bands,2,7,7])                # dim 1: uncorr and corr values
    p_krusk_bdnf_pn  = np.zeros([N_bands,2,7,7])
    
    for fb in range(N_bands):
        for n1 in range(7):
            for n2 in range(7):
                a = np.array([[ np.mean([data1a[f][s][n1][n2] 
                                         for f in f_bands[fb]]) 
                               for s in bdnf_idx[k]] for k in range(2)])
                h,p = st.kruskal(a[0],a[1])
                #h,p = st.mannwhitneyu(a[0],a[1])
                
                h_krusk_bdnf_pn[fb,0,n1,n2] = h
                p_krusk_bdnf_pn[fb,0,n1,n2] = p
        p1  = p_krusk_bdnf_pn[fb][0]
        ixu = np.triu_indices(7)
        p2  = p1[ixu]
        corrv = (multicomp.multipletests(p2,method=corr_method)[1])    
        for iu in range(28):
            p_krusk_bdnf_pn[fb,1,ixu[0][iu],ixu[1][iu]] = corrv[iu]
            p_krusk_bdnf_pn[fb,1,ixu[1][iu],ixu[0][iu]] = corrv[iu]
    
    
    
    # plot sign BDNF differences  
    fig,ax = plt.subplots(N_bands,2,figsize=[10,12])
    for fb in range(N_bands):
        for j in range(2):
            im=ax[fb][j].imshow( (p_krusk_bdnf_pn[fb][j]<0.05),
                                vmin=0,vmax=1,interpolation='none',origin='bottom',cmap='Reds')
            ax[fb][j].grid(linestyle='')
            ax[fb][j].set_xticks(np.arange(0,6.1,1))
            ax[fb][j].set_yticks(np.arange(0,6.1,1))
            ax[fb][j].set_xticklabels(network_names);
            ax[fb][j].set_yticklabels(network_names);
            fig.colorbar(im, ax=ax[fb][j])
        ax[fb][j].set_ylabel(f_band_str[fb])   
     
    return fig
    
    
    


        
     