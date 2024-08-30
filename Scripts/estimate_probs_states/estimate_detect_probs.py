#data format library
import h5py

#numpy
import numpy as np
import numpy.ma as ma
import sys
sys.path.append('/Markov_Fish/utils/') ## Update path here
import os
import copy
import clustering_methods as cl
import operator_calculations as op_calc
import delay_embedding as embed
import stats
import time
import argparse
from joblib import Parallel, delayed

def get_bins(epsilon,r_max):
    xrange = np.arange(-r_max,r_max+epsilon,epsilon)
    yrange = np.arange(-r_max,r_max+epsilon,epsilon)
    centers_x = (xrange[1:]+xrange[:-1])/2
    centers_y = (yrange[1:]+yrange[:-1])/2
    n_bins = len(centers_x)
    rads = np.zeros((n_bins,n_bins))
    for kx,x in enumerate(centers_x):
        for ky,y in enumerate(centers_y):
            rads[kx,ky] = np.sqrt(x**2+y**2)
    return xrange,yrange,rads


def rot(theta):
    mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return mat


def main(argv):
    t_0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-len_sim','--L',help="len_sim",default=1000,type=int)
    parser.add_argument('-idx','--Idx',help="idx",default=0,type=int)
    parser.add_argument('-n_bouts','--N',help="n_bouts",default=1000,type=int)
    args=parser.parse_args()
    
    ms,l_idx = np.array(np.loadtxt('/Markov_Fish/Scripts/estimate_probs_states/iteration_indices_l_detect.txt')[args.Idx],dtype=int) ##Update Path here
    print('ms={},l_idx={}'.format(ms,l_idx),flush=True)
    
    l_range = [.2,.4,.6]
    
    len_sim = args.L
    n_bouts = args.N

    path_to_filtered_data = '/Users/gautam.sridhar/Documents/Repos/Markov_Fish/Datasets/JM_Data/'

    Xs=[]
    for idx in range(100):
        f = h5py.File(path_to_filtered_data + '/zebrafish_ms_sims/sims_lensim_{}_state_{}_{}.h5'.format(len_sim,ms,idx),'r')
        X_sims = np.array(f['sims_X'])
        f.close()
        Xs.append(X_sims)
    Xs = np.concatenate(Xs,axis=0)
        
    n_sims = Xs.shape[0]

    #body length_regime - for exploration
    rmin=0.8
    maxR = 100.
    dr= l_range[l_idx]
    xrange,yrange,rads = get_bins(dr,maxR)
    dist_range = np.logspace(np.log10(rmin),np.log10(maxR),50)
    print(len(dist_range))  
        
       
    print("Estimate probs",flush=True)
    
    print('n_sims={},n_bouts={},dr={:.2f}'.format(n_sims,n_bouts,dr),flush=True)

    ac_samples= np.zeros((n_sims,len(dist_range)))
    for ksim in range(n_sims):
        xy_all = Xs[ksim]
        freqs,_,_= np.histogram2d(xy_all[:,0],xy_all[:,1],bins=[xrange,yrange])
        for kd,ds in enumerate(dist_range):
            sel = rads<=ds
            ac_samples[ksim,kd] = (freqs.T[sel]>0).sum()/sel.sum()  
        if ksim%10==0:
            print(ksim,flush=True)


    
    print(ac_samples.shape,flush=True)
    
    print("Save probs",flush=True)

    f = h5py.File(path_to_filtered_data + 'zebrafish_ms_sims/est_probs/detect_probs_state_{}_nbouts_{}_l_{:.2f}.h5'.format(ms,n_bouts,dr),'w')
    s_ = f.create_dataset('ac_samples',ac_samples.shape)
    s_[...] = ac_samples
    f.close()
    t_f = time.time()
    print('It took {:.2f} minutes.'.format((t_f-t_0)/60.))
    
if __name__ == "__main__":
    main(sys.argv)