#data format library
import h5py

#numpy
import numpy as np
import pandas as pd
import numpy.ma as ma
import sys
sys.path.append('/Users/gautam.sridhar/Documents/Repos/ZebraBouts/utils/')
import os
import copy
import clustering_methods as cl
import operator_calculations as op_calc
import delay_embedding as embed
import stats
import time
import scipy
from joblib import Parallel, delayed
import msmtools.estimation as msm_estimation
from scipy.sparse import diags,identity,coo_matrix, csr_matrix



def unwrapma(x):
    # Adapted from numpy unwrap, this version ignores missing data
    idx = ma.array(np.arange(0,x.shape[0]), mask=x.mask)
    idxc = idx.compressed()
    xc = x.compressed()
    dd = np.diff(xc)
    ddmod = np.mod(dd+np.pi, 2*np.pi)-np.pi
    ddmod[(ddmod==-np.pi) & (dd > 0)] = np.pi
    phc_correct = ddmod - dd
    phc_correct[np.abs(dd)<np.pi] = 0
    ph_correct = np.zeros(x.shape)
    ph_correct[idxc[1:]] = phc_correct
    up = x + ph_correct.cumsum()
    return up


def rec_trajectory(dpsi,dist,psi0,X0):
    rec_psis = psi0+np.cumsum(dpsi)
    rec_vecs = ma.vstack([dist[1:]*np.cos(rec_psis[:-1]),dist[1:]*np.sin(rec_psis[:-1])]).T
    rec_X_traj = X0+np.cumsum(rec_vecs,axis=0)
    return np.vstack([X0,rec_X_traj])


def rec_trajectory_reflective(dpsi,dist,psi0,X0,xmin,xmax):
    psi=psi0
    X = X0
    Xs=[X,]
    for k in range(len(dpsi)-1):
        psi = psi + dpsi[k]
        vec = np.array([dist[k+1]*np.cos(psi),dist[k+1]*np.sin(psi)])
        newX = X+vec
        if np.any(newX>xmax) or np.any(newX<xmin):
            if newX[0]>xmax:
                normal_vec = np.array([-1,0])
            elif newX[0]<xmin:
                normal_vec = np.array([1,0])
            elif newX[1]>xmax:
                normal_vec = np.array([0,-1])
            elif newX[1]<xmin:
                normal_vec = np.array([0,1])
            new_vec = vec-2*vec.dot(normal_vec)*normal_vec
            newX = X+new_vec
            if np.any(newX>xmax) or np.any(newX<xmin):
                print('!',X,X+vec,newX)
                if newX[0]>xmax:
                    newX[0] = 2*xmax-newX[0]
                elif newX[0]<xmin:
                    newX[0] = 2*xmin-newX[0]
                elif newX[1]>xmax:
                    newX[1] = 2*xmax-newX[1]
                elif newX[1]<xmin:
                    newX[1] = 2*xmin-newX[1]
                print('fixed',newX)
            new_vec = newX-X
            psi = np.arctan2(new_vec[1],new_vec[0])
        X = newX
        Xs.append(X)
    return np.vstack(Xs)



def MSD_unc(x, lags):
    mu = ma.zeros((len(lags),))
    Unc = ma.zeros((len(lags),))
    Unc.mask=True
    for i, lag in enumerate(lags):
        if lag==0:
            mu[i] = 0
            Unc[0]=len(x[:,0].compressed())
        elif lag >= x.shape[0]:
            mu[i] = ma.masked
        else:
            x0 = x[lag:,:]
            x1 = x[:-lag,:]
            displacements = ma.sum((x0 - x1)**2,axis=1)
            mu[i] = displacements.mean()
            Unc[i]=len(displacements.compressed())
    return mu,Unc


def simulate(P,state0,iters,lcs):
    states = np.zeros(iters,dtype=int)
    states[0]=state0
    state=state0
    for k in range(1,iters):
        new_state = np.random.choice(np.arange(P.shape[1]),p=list(np.hstack(P[state,:].toarray())))
        state=new_state
        states[k]=state
    return lcs[states]


def simulate_parallel(P,state0,len_sim,lcs):
    return simulate(P,state0,len_sim,lcs)


def get_sims_ensemble(labels_all, P_ensemble, delay=1, n_sims=10, len_sim=1000):
    """
    Generate simulations from the ensemble operator
    """
    labels_state0 = labels_all.copy()
    
    lookup_table = {}
    for state in np.unique(labels_state0.compressed()):
        mask = labels_state0==state
        lookup_table[state] = np.arange(len(labels_state0))[mask]
        
    lcs_ensemble = msm_estimation.largest_connected_set(P_ensemble)
    print(lcs_ensemble.shape)
    final_labels_state = op_calc.get_connected_labels(labels_state0,lcs_ensemble)
    states0 = np.ones(n_sims,dtype=int)*final_labels_state.compressed()[0]
    sims = Parallel(n_jobs=100)(delayed(simulate_parallel)(P_ensemble,state0,len_sim,lcs_ensemble) for state0 in states0)
    
    return sims


def get_sims_state(state_idx,labels_fish,kmeans_labels,psi_start, 
                   dpsi,dist,X_start,delay=1,n_sims=10, len_sim = 1000):
    labels_state0 = ma.hstack(labels_fish).copy()
    labels_state0[kmeans_labels!=state_idx] = ma.masked
    labels_state0_rec = labels_state0.reshape(labels_fish.shape)

    labels_state0 = ma.hstack(labels_state0_rec.copy())

    lookup_table = {}
    for state in np.unique(labels_state0.compressed()):
        mask = labels_state0==state
        lookup_table[state] = np.arange(len(labels_state0))[mask]

    lcs,P = op_calc.transition_matrix(labels_state0,delay,return_connected=True)
    final_labels_state = op_calc.get_connected_labels(labels_state0,lcs)
    unique_labels = np.unique(final_labels_state.compressed())
    states0 = np.random.choice(unique_labels,n_sims).astype(int)
    # states0 = np.ones(n_sims,dtype=int)*final_labels_state.compressed()[0]
    sims = Parallel(n_jobs=100)(delayed(simulate_parallel)(P,state0,len_sim,lcs) for state0 in states0)
    
    sims_X_traj = []
    for sim in sims:
        dpsi_sim = np.array([dpsi[lookup_table[s][np.random.randint(0,len(lookup_table[s]))]] for s in sim])
        dist_sim = np.array([dist[lookup_table[s][np.random.randint(0,len(lookup_table[s]))]] for s in sim])
        sim_X_traj = rec_trajectory(dpsi_sim,dist_sim,psi_start,X_start)
        sims_X_traj.append(ma.array(sim_X_traj))
    return sims,sims_X_traj


def get_sims_group(labels_group,P,dpsi,dist,psi_start,X_start,delay=1,n_sims=10, len_sim = 1000):

    lookup_table = {}
    for state in np.unique(labels_group.compressed()):
        mask = labels_group==state
        lookup_table[state] = np.arange(len(labels_group))[mask]
    
    lcs,P = op_calc.transition_matrix(labels_group,delay,return_connected=True)
    final_labels_state = op_calc.get_connected_labels(labels_group,lcs)
    unique_labels = np.unique(final_labels_state.compressed())
    states0 = np.random.choice(unique_labels,n_sims).astype(int)
    
    sims = Parallel(n_jobs=100)(delayed(simulate_parallel)(P,state0,len_sim,lcs) for state0 in states0)
    
    sims_X_traj = []
    for sim in sims:
        dpsi_sim = np.array([dpsi[lookup_table[s][np.random.randint(0,len(lookup_table[s]))]] for s in sim])
        dist_sim = np.array([dist[lookup_table[s][np.random.randint(0,len(lookup_table[s]))]] for s in sim])
        sim_X_traj = rec_trajectory(dpsi_sim,dist_sim,psi_start,X_start)
        sims_X_traj.append(ma.array(sim_X_traj))
    return sims,sims_X_traj


def get_sims_group2(labels_fish,lookup_table,fish_idx,dpsi,dist,psi_start,X_start,delay=1,n_sims=10, len_sim = 1000):
    labels_group = labels_fish.copy()
    labels_group[fish_idx] = ma.masked
    labels_group = ma.hstack(labels_group)

    lcs,P = op_calc.transition_matrix(labels_group,delay,return_connected=True)
    final_labels_state = op_calc.get_connected_labels(labels_group,lcs)
    unique_labels = np.unique(final_labels_state.compressed())
    states0 = np.random.choice(unique_labels,n_sims).astype(int)
    
    sims = Parallel(n_jobs=100)(delayed(simulate_parallel)(P,state0,len_sim,lcs) for state0 in states0)
    
    sims_X_traj = []
    for sim in sims:
        dpsi_sim = np.array([dpsi[lookup_table[s][np.random.randint(0,len(lookup_table[s]))]] for s in sim])
        dist_sim = np.array([dist[lookup_table[s][np.random.randint(0,len(lookup_table[s]))]] for s in sim])
        sim_X_traj = rec_trajectory(dpsi_sim,dist_sim,psi_start,X_start)
        sims_X_traj.append(ma.array(sim_X_traj))
    return sims,sims_X_traj


def get_sims_group3(labels_fish,P,lookup_table,dpsi,dist,psi_start,X_start,delay=1,n_sims=10, len_sim = 1000):
    P = csr_matrix(P)
    lcs = msm_estimation.largest_connected_set(P)
    labels_group = labels_fish.copy()
    labels_group = ma.hstack(labels_group)

    final_labels_state = op_calc.get_connected_labels(labels_group,lcs)
    unique_labels = np.unique(final_labels_state.compressed())
    states0 = np.random.choice(unique_labels,n_sims).astype(int)
    
    sims = Parallel(n_jobs=100)(delayed(simulate_parallel)(P,state0,len_sim,lcs) for state0 in states0)
    
    sims_X_traj = []
    for sim in sims:
        dpsi_sim = np.array([dpsi[lookup_table[s][np.random.randint(0,len(lookup_table[s]))]] for s in sim])
        dist_sim = np.array([dist[lookup_table[s][np.random.randint(0,len(lookup_table[s]))]] for s in sim])
        sim_X_traj = rec_trajectory(dpsi_sim,dist_sim,psi_start,X_start)
        sims_X_traj.append(ma.array(sim_X_traj))
    return sims,sims_X_traj



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



























