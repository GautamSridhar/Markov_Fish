#data format library
import h5py

#numpy
import numpy as np
import numpy.ma as ma
import sys
sys.path.append('/home/a/antonio-costa/Zebrafish_ms_sims/utils/')
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


def get_bins_angs(epsilon,r_max):
    xrange = np.arange(-r_max,r_max+epsilon,epsilon)
    yrange = np.arange(-r_max,r_max+epsilon,epsilon)
    centers_x = (xrange[1:]+xrange[:-1])/2
    centers_y = (yrange[1:]+yrange[:-1])/2
    n_bins = len(centers_x)
    rads = np.zeros((n_bins,n_bins))
    angs = np.zeros((n_bins,n_bins))
    for kx,x in enumerate(centers_x):
        for ky,y in enumerate(centers_y):
            rads[kx,ky] = np.sqrt(x**2+y**2)
    for i,y in enumerate(centers_y[::-1]):
        for j,x in enumerate(centers_x):
            rads[i,j] = np.sqrt(x**2+y**2)
            angs[i,j] = np.arctan2(y,x)
    return xrange,yrange,rads, angs

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
    
    ms,l_idx = np.array(np.loadtxt('/home/a/antonio-costa/Zebrafish_ms_sims/estimate_probs_states/iteration_indices_l_hunt.txt')[args.Idx],dtype=int)
    print('ms={},l_idx={}'.format(ms,l_idx),flush=True)
    
    l_range = [.01,.02,.03,.04]
    
    len_sim = args.L
    n_bouts = args.N

    Xs=[]
    for idx in range(100):
        f = h5py.File('/flash/StephensU/antonio/zebrafish_ms_sims/sims_lensim_{}_state_{}_{}.h5'.format(len_sim,ms,idx),'r')
        X_sims = np.array(f['sims_X'])
        f.close()
        Xs.append(X_sims)
    Xs = np.concatenate(Xs,axis=0)
        
    n_sims = Xs.shape[0]

    rmin=0.0
    maxR = .2
    dr= l_range[l_idx]
    print('r={:.1f}, maxR = {:.1f},dr = {:.2f}'.format(rmin,maxR,dr),flush=True)
    xrange,yrange,rads,angs = get_bins_angs(dr,maxR)
    psi_range = np.array([-np.pi/6, np.pi/6])
    sel1 = angs >= psi_range.min()
    sel2 = angs <= psi_range.max()
    sel_phi=  np.logical_and(sel1,sel2)
    sel_rad = rads<maxR
    sel = np.logical_and(sel_phi,sel_rad)    
        
       
    print("Estimate probs",flush=True)
    
    print('n_sims={},n_bouts={}'.format(n_sims,n_bouts),flush=True)

    ac_samples= np.zeros((n_sims,))
    for ksim in range(n_sims):
        xy_all = Xs[ksim]
        vec0 = xy_all[1] - xy_all[0]
        psi0 = np.arctan2(vec0[1],vec0[0])
        xy_all = xy_all[1:]-xy_all[1]
        xy_all_ = np.dot(xy_all,rot(psi0))
        freqs,bins_x,bins_y= np.histogram2d(xy_all_[:n_bouts,0],xy_all_[:n_bouts,1],bins=[xrange,yrange])
        ac_samples[ksim] = (freqs.T[sel]>0).sum()/sel.sum()  
        if ksim%10==0:
            print(ksim,flush=True)


    
    print(ac_samples.shape,flush=True)
    
    print("Save probs",flush=True)

    f = h5py.File('/flash/StephensU/antonio/zebrafish_ms_sims/est_probs/hunt_probs_state_{}_nbouts_{}_l_{:.2f}.h5'.format(ms,n_bouts,dr),'w')
    s_ = f.create_dataset('ac_samples',ac_samples.shape)
    s_[...] = ac_samples
    f.close()
    t_f = time.time()
    print('It took {:.2f} minutes.'.format((t_f-t_0)/60.))
    
if __name__ == "__main__":
    main(sys.argv)