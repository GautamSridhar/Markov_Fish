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
from scipy.sparse import diags,identity,coo_matrix, csr_matrix
import msmtools.estimation as msm_estimation

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

def rec_trajectory(dpsi,dist,psi0,X0):
    rec_psis = psi0+np.cumsum(dpsi)
    rec_vecs = ma.vstack([dist[1:]*np.cos(rec_psis[:-1]),dist[1:]*np.sin(rec_psis[:-1])]).T
    rec_X_traj = X0+np.cumsum(rec_vecs,axis=0)
    return np.vstack([X0,rec_X_traj])


def get_sims_state(state_idx,labels_fish,kmeans_labels,psi_start,dpsi,dist,X_start,delay=1,n_sims=10, len_sim = 1000):
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
    sims = Parallel(n_jobs=100)(delayed(simulate_parallel)(P,state0,len_sim,lcs) for state0 in states0)
    
    sims_X_traj = []
    for sim in sims:
        dpsi_sim = np.array([dpsi[lookup_table[s][np.random.randint(0,len(lookup_table[s]))]] for s in sim])
        dist_sim = np.array([dist[lookup_table[s][np.random.randint(0,len(lookup_table[s]))]] for s in sim])
        sim_X_traj = rec_trajectory(dpsi_sim,dist_sim,psi_start,X_start)
        sims_X_traj.append(ma.array(sim_X_traj))
    return sims,sims_X_traj


def main(argv):
    t_0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-len_sim','--L',help="len_sim",default=1000,type=int)
    parser.add_argument('-idx','--Idx',help="idx",default=0,type=int)
    args=parser.parse_args()
    idx = args.Idx

    ms,idx = np.array(np.loadtxt('/Markov_Fish/Scripts/sims/iteration_indices_strats.txt')[idx],dtype=int) ##Update path here
    print('ms={},idx={}'.format(ms,idx),flush=True)
    
    
    print("Load data",flush=True)
    path_to_filtered_data = '/Users/gautam.sridhar/Documents/Repos/Markov_Fish/Datasets/JM_Data/'
    
    f = h5py.File(path_to_filtered_data + 'kmeans_labels_K5_N1200_s8684.h5','r')
    lengths_all = np.array(f['MetaData/lengths_data'], dtype=int)
    labels_fish = ma.array(f['labels_fish'],dtype=int)
    state_trajs = ma.array(f['state_trajs'])
    f.close()
    to_mask = 1300
    labels_fish[labels_fish == to_mask] = ma.masked
    labels_all= ma.concatenate(labels_fish,axis=0)
    
    f = h5py.File(path_to_filtered_data+'zebrafish_ms_sims/X_pos.h5','r')
    X = ma.array(f['X'])
    X[X==0] = ma.masked
    X_head = ma.array(f['X_head'])
    X_head[X_head==0]=ma.masked
    f.close()
    
    vecX = ma.diff(X[:,0,:],axis=0)
    dist = ma.zeros(X.shape[0])
    dist[:-1] = ma.sqrt(vecX[:,0]**2+vecX[:,1]**2)
    dist[-1] = ma.masked 

    psi = ma.zeros(X.shape[0])
    psi[:-1] = ma.arctan2(vecX[:,1],vecX[:,0])
    psi[-1] = ma.masked

    psi_unwrap = unwrapma(psi)
    dpsi = ma.zeros(X.shape[0])
    dpsi[:-1] = psi_unwrap[1:]-psi_unwrap[:-1]
    dpsi[-1:] = ma.masked

    psi_fish = psi.reshape((X_head[:].shape[0], X_head[:].shape[1]))
        
    lookup_table = {}
    for state in np.unique(labels_all.compressed()):
        mask = labels_all==state
        lookup_table[state] = np.arange(len(labels_all))[mask]
    
    
    P_ensemble = np.load(path_to_filtered_data + 'P_ensemble_ex8_N1200_s8684.npy')


    P_ensemble = csr_matrix(P_ensemble)
    
    delay = 3
    dt = 1
    lcs_ensemble = msm_estimation.largest_connected_set(P_ensemble)
    inv_measure = op_calc.stationary_distribution(P_ensemble)
    final_labels = op_calc.get_connected_labels(labels_all,lcs_ensemble)
    
    kms2 = np.load(path_to_filtered_data + 'cg4_labels.npy')
    final_labels_recs = final_labels.reshape(labels_fish.shape)
    kmeans_labels_traj = ma.masked_invalid(kms2[final_labels])
    kmeans_labels_traj[final_labels.mask] = ma.masked
    
    cluster_traj_all = ma.copy(final_labels)
    cluster_traj_all[~final_labels.mask] = ma.array(kms2)[final_labels[~final_labels.mask]]
    cluster_traj_all[final_labels.mask] = ma.masked

    cluster_fish = cluster_traj_all.reshape(labels_fish.shape[0],labels_fish.shape[1])
    cluster_fish_mask = cluster_fish.mask
    
    print("Run simulations",flush=True)

    n_sims = 100
    len_sim = args.L
 
    sim,sim_X_traj = get_sims_state(ms,labels_fish,kmeans_labels_traj,np.random.choice(psi_fish[:,0]), dpsi, dist, [0,0],
                                     n_sims=n_sims, len_sim=len_sim)

        
    sims = np.array(sim)
    sims_X_traj = np.array(sim_X_traj)
    
    print(sims.shape,flush=True)
    print(sims_X_traj.shape,flush=True)
    
    print("Save simulations",flush=True)

    f = h5py.File(path_to_filtered_data + 'zebrafish_ms_sims/sims_lensim_{}_state_{}_{}.h5'.format(len_sim,ms,idx),'w')
    s_ = f.create_dataset('sims',sims.shape)
    s_[...] = sims
    sX_ = f.create_dataset('sims_X',sims_X_traj.shape)
    sX_[...] = sims_X_traj
    f.close()
    t_f = time.time()
    print('It took {:.2f} minutes.'.format((t_f-t_0)/60.))
    
if __name__ == "__main__":
    main(sys.argv)