# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 19:46:31 2017

@author: Felix Siebenhühner

partially adapted from astropy module




"""
import numpy as np
import scipy
import statsmodels.sandbox.stats.multicomp as mc
import matplotlib.pyplot as plt
import scipy.stats as stats

#%% private

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



#%% misc

def peak_finder(freqs,values,fmin=5,fmax=14,fstep=0.1):
    '''
    simple code to find the peak in values for a narrow range of frequencies, 
    using interpolation,  we should add detrending (e.g. 1/f)
    
    INPUT:
        freqs:     list of frequencies used in orig. data
        values:    array [groups x values]
        fmin,fmax: in Hz
        fstep:     in Hz
    OUTPUT:
        peak_freqs: list of peak freqs. for each group
        x_vals:     1D array of x values for the fits, i.e. frequencies
        y_vals:     2D array of y values [groups x frequencies]
    
    '''    

    N_gr        = len(values)                         #values in format groups x values
    y_vals      = np.zeros([N_gr,int((fmax-fmin)/fstep)])
    peak_freqs  = np.zeros(N_gr)
    x_vals      = np.arange(fmin,fmax,fstep)

    for r in range(N_gr):
        f = scipy.interpolate.interp1d(freqs,values[r],kind='cubic')
        y = f(x_vals)
        y_vals[r] = y
        peak_freqs[r]=x_vals[np.argmax(y)]
        
    return peak_freqs, x_vals, y_vals    




#%% permutation testing



def prox_cluster_1D(sigs):
    N_obs = len(sigs)
    clusters = []
    c = 0
    for i in range(N_obs):
        if sigs[i]:
            if len(clusters)==0:
                clusters.append([i])                
            elif (i-1 in clusters[c]):
                clusters[c].append(i)
            else:
                c += 1
                clusters.append([i]) 
    return clusters



def CI_from_bootstrap(data, bootnum=1000, low=2.5, high=97.5, N_array=None, samples=None, bootfunc=None):
         
    '''adapted from astropy module
       data can have any number ND of dimensions, bootstrapping will be done for the 0th dimension
       N_array can be used for weighing, can have 1 or ND-1 dimensions 
       e.g. shape(data) = 5x30x60   --> shape(N_array)=30 or shape(N_array)=30x60        
       
    '''
    
    if samples is None:
        samples = data.shape[0]
    
    # make sure the input is sane
    if samples < 1 or bootnum < 1:
        raise ValueError("neither 'samples' nor 'bootnum' can be less than 1.")
    
    if bootfunc is None:
        resultdims = (bootnum,) + (samples,) + data.shape[1:]
    else:
    # test number of outputs from bootfunc, avoid single outputs which are
    # array-like
        try:
            resultdims = (bootnum, len(bootfunc(data)))
        except TypeError:
            resultdims = (bootnum,)
        
    # create empty boot array
    boot = np.empty(resultdims)
    
    for i in range(bootnum):
        bootarr = np.random.randint(low=0, high=data.shape[0], size=samples)
        if bootfunc is None:
            boot[i] = data[bootarr]
        else:
            boot[i] = bootfunc(data[bootarr])
        if N_array is not None:
            N = np.nansum(N_array[bootarr],0)
            N.astype(float)
            boot[i] = boot[i]/N
    
    if N_array is not None:
        mean      = np.nansum(data,0)/np.nansum(N_array,0)
        means     = np.nansum(boot,1)
    else: 
        mean      = np.nanmean(data,0)
        means     = np.nanmean(boot,1)
        
    mean_boot = np.nanmean(means,0)
    lower     = np.percentile(means,low,0)
    upper     = np.percentile(means,high,0)
    
    return mean, lower, upper, mean_boot



     




#%% CORRELATION FUNCTIONS


def correlate(values, scores, method='Spearman', remove_nan = True, print_N = False):
    
    ''' 
    INPUT:
        values: 1D array
        scores: 1D array
    OUTPUT:
        corr, p_val: single values
    '''
    
    if remove_nan:
        inds1 = np.where(values != values)[0]
        inds2 = np.where(scores != scores)[0]
        inds  = [i for i in range(len(values)) if not i in inds1 and i not in inds2]
        values  = values[inds]
        scores  = scores[inds]
        if print_N:
            print(len(values))

    if method=='Pearson':
        corr, p_val = scipy.stats.pearsonr(values,scores)
    elif method=='Spearman':
        corr, p_val = scipy.stats.spearmanr(values,scores)  
        
    return corr, p_val   


def correlate_p(values, scores, method='Spearman'):
    ''' 
    INPUT:
        values: 2D array, size [sets x parcels]
        score:  either 1D aray [sets] or 2D array [sets x parcels]
    OUTPUT:
        corrs_p, p_vals_p: 1D arrays [parcels]
    '''
        
    N_parc     = len(values[0])  
    corrs_p    = np.zeros([N_parc])
    p_vals_p   = np.zeros([N_parc])  
    
    if _depth(scores) == 1:
        scores = np.transpose(np.tile(scores,(N_parc,1)))
        
    for p in range(N_parc):
        if method=='Pearson':
            corrs_p[p], p_vals_p[p] = scipy.stats.pearsonr(values[:,p],scores[:,p])
        elif method=='Spearman':
            corrs_p[p], p_vals_p[p] = scipy.stats.spearmanr(values[:,p],scores[:,p]) 
            
    return corrs_p, p_vals_p   


def correlate_matrix(values1,values2,method='Spearman',remove_nan=True):
    ''' 
    INPUT:
        values1: 2D array, size [parcels x parcels]
        values2: 2D array, size [parcels x parcels]
    OUTPUT:
        corr, p_val:  scalars
    '''
    
    N_pix = np.product(values1.shape)
    val1r = np.reshape(values1,N_pix)
    val2r = np.reshape(values2,N_pix)
    
    if remove_nan:
        inds1 = np.where(val1r != val1r)[0]
        inds2 = np.where(val2r != val2r)[0]
        inds = [i for i in range(N_pix) if not i in inds1 and i not in inds2]
        val1r  = val1r[inds]
        val2r  = val2r[inds]
        
    if method=='Pearson':
            corr, p_val = scipy.stats.pearsonr(val1r,val2r)
    elif method=='Spearman':
            corr, p_val = scipy.stats.spearmanr(val1r,val2r) 
            
    return corr, p_val
            

def correlate_matrix_and_plot(values1,values2,method='Spearman',xlab='',ylab='',
                              remove_nan=True,title='corrs'):
    ''' 
    INPUT:
        values1: 2D array, size [parcels x parcels]
        values2: 2D array, size [parcels x parcels]
    OUTPUT:
        corr, p_val:  scalars
    '''
    
    N_parc = len(values1)
    val1r = np.reshape(values1,[N_parc**2])
    val2r = np.reshape(values2,[N_parc**2])
    
    if remove_nan:
        inds1 = np.where(val1r != val1r)[0]
        inds2 = np.where(val2r != val2r)[0]
        inds = [i for i in range(N_parc**2) if not i in inds1 and i not in inds2]
        val1r  = val1r[inds]
        val2r  = val2r[inds]
        
    if method=='Pearson':
            corr, p_val = scipy.stats.pearsonr(val1r,val2r)
    elif method=='Spearman':
            corr, p_val = scipy.stats.spearmanr(val1r,val2r) 
            
    plt.scatter(val1r,val2r)        
    a1,a0  = np.polyfit(val1r,val2r,deg=1)
    xvals2 = np.arange(min(val1r)*0.9,max(val1r)*1.1,0.1)
    plt.plot(xvals2,xvals2*a1+a0,'k')  
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if title=='corrs':
        plt.title(method+' r='+str(np.around(corr,3))+', p='+str(np.around(p_val,3))+'\n')
    else:
        plt.title(title)
            
    return corr, p_val



def correlate_3D(values1,values2, method='Spearman',show_f=False):   
    ''' 
    Correlates two arrays along the first dimension.
    INPUT:
        values: 3D array [sets x dim1 x dim2] 
        scores: 1D array [sets] or
                3D array [sets x dim1 x dim2]
    OUTPUT:
        corrs_e, p_vals_e : 2D arrays [dim1 x dim2]
    '''   
    
    dim1      = len(values1[0])      
    dim2      = len(values1[0][0])
    corrs_e   = np.zeros([dim1,dim2])
    p_vals_e  = np.zeros([dim1,dim2])  
    
    if _depth(values2) == 1:
        values2 = np.swapaxes(np.tile(values2,(dim2,dim1,1)),0,2)
    
    for p1 in range(dim1):
        for p2 in range(dim2):
            if method == 'Pearson':
                dummy = scipy.stats.pearsonr (values1[:,p1,p2],values2[:,p1,p2])
            elif method=='Spearman':
                dummy = scipy.stats.spearmanr(values1[:,p1,p2],values2[:,p1,p2])
            corrs_e [p1,p2] = dummy[0]
            p_vals_e[p1,p2] = dummy[1]

    return corrs_e, p_vals_e



def correlate_and_plot_2groups(scores,values,indsA,gr_names,score_name,val_name,
                               method='Spearman',title='corrs',colors=['r','b']):
    r,p = [np.around(i,3) for i in correlate(scores,values,method)]    
    plt.figure(figsize=[4,3])
    for i in range(2):
        inds = indsA[i]
        plt.scatter(scores[inds],values[inds],c=colors[i])        
    plt.legend(gr_names)     
    sc0 = scores
    ix = np.where(sc0==sc0)
    a1,a0=np.polyfit(scores[ix],values[ix],deg=1)
    xvals2 = np.arange(min(sc0)*0.9,max(sc0)*1.1,0.1)   
    plt.plot(xvals2,xvals2*a1+a0,'k')  
    plt.xlabel(score_name)
    plt.ylabel(val_name)
    if title=='corrs':
        plt.title(method+' r='+str(r)+' p='+str(p)+'\n')
    else:
        plt.title(title)
    plt.show()



# def correlate_along(values1,values2, method='Spearman',axis,show_f=False):   
#     ''' comput correlation along a specific axis
    
    
#     NEEDS TO BE IMPLEMENTED!
    
    
#     INPUT:
#         values: nD array [sets x parcels x parcels] 
#         scores: nD array [sets x parcels x parcels]
        
#     OUTPUT:
#         corrs_e, p_vals_e : 2D arrays [parcels x parcels]
#     '''   
    
#     N_parc      = len(values[0])      
#     corrs_e     = np.zeros([N_parc,N_parc])
#     p_vals_e    = np.zeros([N_parc,N_parc])  
    
 

#     return corrs_e, p_vals_e



def get_correlations(values, scores, method='Spearman'):    
    ''' 
    INPUT
        values: 3D array [sets x freqs x parcels] 
        scores: 1D array [sets]
    OUTPUT:
        corrs,   p_vals:   1D arrays [freqs]
        corrs_p, p_vals_p: 2D arrays [freqs x parcels]
    '''
    
    N_freq     = len(values[0])
    N_parc     = len(values[0][0])      
    corrs      = np.zeros([N_freq])
    p_vals     = np.zeros([N_freq])
    corrs_p    = np.zeros([N_freq,N_parc])
    p_vals_p   = np.zeros([N_freq,N_parc])       
    mean_op    = np.mean(values,2)
 
    for f in range(N_freq):  
        if method=='Pearson':
            corrs[f],p_vals[f] = scipy.stats.pearsonr(mean_op[:,f],scores)
        else:
            corrs[f],p_vals[f] = scipy.stats.spearmanr(mean_op[:,f],scores)            
        dummy = np.zeros([N_parc,2])
        for p in range(N_parc):
            if method == 'Pearson':
                dummy = scipy.stats.pearsonr (values[:,f,p],scores)
            else:
                dummy = scipy.stats.spearmanr(values[:,f,p],scores)
            corrs_p [f,p] = dummy[0]
            p_vals_p[f,p] = dummy[1]

    return corrs, p_vals, corrs_p, p_vals_p


def get_correlations2(values, scores, method='Spearman'):    
    ''' 
    INPUT:
        values: list of 2D arrays [freqs] x [sets x parcels] 
        scores: 1D array [sets]
    OUTPUT:
        corrs,   p_vals:   1D arrays [freqs]
        corrs_p, p_vals_p: 2D arrays [freqs x parcels]        
    '''
    
    N_freq      = len(values)
    N_parc      = len(values[0][0])      
    corrs       = np.zeros([N_freq])
    p_vals      = np.zeros([N_freq])
    corrs_p     = np.zeros([N_freq,N_parc])
    p_vals_p    = np.zeros([N_freq,N_parc])       
 
    for f in range(N_freq):      
        mean_op     = np.mean(values[f],1)
        if method=='Pearson':
            corrs[f],p_vals[f] = scipy.stats.pearsonr(mean_op,scores)
        else:
            corrs[f],p_vals[f] = scipy.stats.spearmanr(mean_op,scores)            
        dummy = np.zeros([N_parc,2])
        for p in range(N_parc):
            if method == 'Pearson':
                dummy = scipy.stats.pearsonr (values[f][:,p],scores)
            else:
                dummy = scipy.stats.spearmanr(values[f][:,p],scores)
            corrs_p [f,p] = dummy[0]
            p_vals_p[f,p] = dummy[1]

    return corrs, p_vals, corrs_p, p_vals_p





def get_correlations_e(values,scores, method='Spearman',show_f=False):    
    ''' 
    INPUT:
        values: 4D array [sets x freqs x parcels x parcels] 
    OUTPUT:
        corrs_e, p_vals_e: 3D arrays [freqs x parcels x parcels]
    '''
    
    N_freq      = len(values[0])
    N_parc      = len(values[0][0])      
    corrs_e     = np.zeros([N_freq,N_parc,N_parc])
    p_vals_e    = np.zeros([N_freq,N_parc,N_parc])  
    
    for f in range(N_freq):           
        for p1 in range(N_parc):
            for p2 in range(N_parc):
                try:
                    vals = values[:,f,p1,p2]
                    inds = np.where(vals==vals)
                    if method == 'Pearson':
                        dummy = scipy.stats.pearsonr (vals[inds],scores[inds])
                    else:
                        dummy = scipy.stats.spearmanr(vals[inds],scores[inds])
                except:
                    pass
                corrs_e [f,p1,p2] = dummy[0]
                p_vals_e[f,p1,p2] = dummy[1]
        if f%4==0 and f>0 and show_f:
            print('f='+str(f))
    return corrs_e, p_vals_e




#%% multiple comparisons correction

def mc_array(p_vals,mc_meth):
    
    p_vals1 = np.reshape(p_vals, np.prod(p_vals.shape))
    sign_corr  = 1.*mc.multipletests(p_vals1, method =mc_meth)[0]    
    return np.reshape(sign_corr,p_vals.shape)




def sig_FDR_Q_all(p_vals,alpha,Q):
    N        = np.prod(p_vals.shape)
    p_vals2  = np.reshape(p_vals,N)
    sig      = p_vals2<alpha    
    K_uncorr = np.sum(sig)/N
    K_exp    = alpha+Q
    K_corr   = np.max([K_uncorr-K_exp,0])
    pctile   = np.percentile(p_vals2,100*(K_corr))
    sig_Q    =  (p_vals2<pctile) * 1
    return np.reshape(sig_Q,p_vals.shape)

def sig_FDR_Q_pf(p_vals,alpha,Q):
    '''
    p_vals as 2D array [freq x parcel] or 3D [freq x parcel x parcel]
    '''
    sig_Q = np.zeros(np.shape(p_vals))
    for i in range(len(p_vals)):
        pv       = p_vals[i]
        N        = np.prod(pv.shape)
        sig      = pv < alpha    
        K_uncorr = np.sum(sig)/N
        K_exp    = alpha+Q
        K_corr   = np.max([K_uncorr-K_exp,0])
        pctile   = np.percentile(pv,100*(K_corr))
        sig_Q[i] = (pv < pctile) * 1
    return np.reshape(sig_Q,p_vals.shape)



#%% group comparison



def compare(data1,test_type='kw'):
    
    ''' 
    Computes statistics if two or more groups are different, removing nans.
    INPUT: 
        data1: 2D list  [groups] x [sets]
        test_type: can be 'mwu','wilc','paired_t','kw'. 
        
        mwu, wilc, paired_t only compare the two first groups
        
    '''
            
    dx = [[d for d in d1 if d==d] for d1 in data1]   # get rid of nans
    N_groups = len(dx)

    if test_type == 'mwu':
        sta, p_val    = stats.mannwhitneyu(dx[0],dx[1]) 
    elif test_type == 'anova':
        sta, p_val    = stats.f_oneway(dx[0],dx[1])
    elif test_type == 'wilc':
        sta, p_val    = stats.wilcoxon(dx[0],dx[1])
    elif test_type == 'paired_t':
        sta, p_val    = stats.ttest_rel(dx[0],dx[1])
    if test_type == 'kw':        
        if N_groups == 2:
            sta, p_val    = stats.kruskal(dx[0],dx[1])          
        elif N_groups == 3:
            sta, p_val    = stats.kruskal(dx[0],dx[1],dx[2])  
        elif N_groups == 4:
            sta, p_val    = stats.kruskal(dx[0],dx[1],dx[2],dx[3])  
        elif N_groups == 5:
            sta, p_val    = stats.kruskal(dx[0],dx[1],dx[2],dx[3],dx[4]) 
        elif N_groups == 6:
            sta, p_val    = stats.kruskal(dx[0],dx[1],dx[2],dx[3],dx[4],dx[5])  

    return sta, p_val




























