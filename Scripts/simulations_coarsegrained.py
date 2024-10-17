#data format library
import h5py
import argparse

#numpy
import numpy as np
import pandas as pd
import numpy.ma as ma
# %matplotlib notebook
import sys
import functools
import concurrent.futures
sys.path.append('/network/lustre/iss02/home/gautam.sridhar/Markov_Fish/utils/')
import matplotlib.colors as pltcolors
import os
import copy
import clustering_methods as cl
import msmtools.estimation as msm_estimation
from sklearn.cluster import KMeans
import operator_calculations as op_calc
import delay_embedding as embed
import stats
from sklearn.decomposition import PCA
from scipy.sparse import diags,identity,coo_matrix, csr_matrix
import time
import scipy

#np.random.seed(42)


def perform_partition(P_ensemble, labels_fish,delay=3, n_state=5):
    """
    Perform partition on ensemble operator and return recentered eigenfunctions and invariant measure
    """

    # Perform coherence based split on whole dataset
    # np.random.seed(25)
    dt = 1
    num_eigfs = int(np.ceil(np.log2(n_state))) + 1
    labels_all= ma.concatenate(labels_fish,axis=0)
    P_ensemble = csr_matrix(P_ensemble)
    lcs_ensemble = msm_estimation.largest_connected_set(P_ensemble)
    inv_measure = op_calc.stationary_distribution(P_ensemble)
    final_labels = op_calc.get_connected_labels(labels_all,lcs_ensemble)
    R = op_calc.get_reversible_transition_matrix(P_ensemble)
    eigvals,eigvecs = op_calc.sorted_spectrum(R,k=num_eigfs,seed=43)
    sorted_indices = np.argsort(eigvals.real)[::-1]
    eigvals = eigvals[sorted_indices][1:].real
    eigvals[np.abs(eigvals-1)<1e-12] = np.nan
    eigvals[eigvals<1e-12] = np.nan
    t_imp =  -(delay*dt)/np.log(np.abs(eigvals))
    eigfunctions = eigvecs.real/np.linalg.norm(eigvecs.real,axis=0)

    split_locs = []
    distorted_eigfs = np.zeros((eigfunctions.shape[0], eigfunctions.shape[1]-1))
    print('Coarse graining', flush=True)
    for i in range(1,eigfunctions.shape[1]):
        print(i, flush=True)
        phi = eigfunctions[:,i]
        _,_,_,split_idx,_ = op_calc.optimal_partition(phi,inv_measure,P_ensemble,return_rho=True)

        sort_range = np.sort(phi)
        neg_range = np.linspace(-1,0, len(sort_range[0:split_idx]))
        pos_range = np.linspace(0,1,len(sort_range[split_idx:]))
        distort_r = np.hstack([neg_range,pos_range])
        distort = np.zeros(phi.shape)

        pos = [np.where(phi == a)[0][0] for a in np.sort(phi)]

        for j in range(phi.shape[0]):
            distort[pos[j]] = distort_r[j]

        distorted_eigfs[:,i-1] = np.sqrt((-1/np.log(eigvals[i-1]))/2)*distort
        split_locs.append(split_idx)

    return distorted_eigfs, inv_measure, final_labels


def coarse_grain(eigfs, final_labels, labels_fish,inv_measure, n_state=2):
    """
    Coarse grain the space according to kmeans-embedding on the recentered eigenfunctions
    and return clustered trajectories
    """
    num_eigfs = int(np.ceil(np.log2(n_state)))
    X = eigfs[:,:num_eigfs]
    km = KMeans(n_clusters=n_state, n_init=1000,random_state=123).fit(X, sample_weight=inv_measure)
    
    cluster_traj_all = ma.copy(final_labels)
    cluster_traj_all[~final_labels.mask] = ma.array(km.labels_)[final_labels[~final_labels.mask]]
    cluster_traj_all[final_labels.mask] = ma.masked

    cluster_fish = cluster_traj_all.reshape(labels_fish.shape[0],labels_fish.shape[1])
    cluster_fish_mask = cluster_fish.mask

    return cluster_fish, cluster_fish_mask


def simulate_fish(delay, params):

    s = params[0]
    fish_num = params[1]
    labels_ = params[2]

    if ma.count(labels_) > 50:
        len_sim = int(len(labels_.compressed())/delay)
        lcs,P = op_calc.transition_matrix(labels_,delay, return_connected=True)
        final_labels = op_calc.get_connected_labels(labels_,lcs)
        state0 = final_labels.compressed()[0]
        print('Starting Fish', fish_num,flush=True)
        sim_states = op_calc.simulate(P,state0, len_sim)
        print('Finished Fish', fish_num, flush=True)

        return s, fish_num, lcs[sim_states]
    else:
        print('Recording {} skipped'.format(fish_num),flush=True)


def main(argv):
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-seeds','--Seeds',help="Number of seeds to evaluate over",type=int,default=1000)
    parser.add_argument('-cg','--Scale',help="Number of metastates to coarse grain into",type=int,default=2)
    parser.add_argument('-t','--Tau', help='Delay of the PF operator', type=int, default=1)
    parser.add_argument('-k','--K',help="Number of delays to choose",default=8, type=int)
    parser.add_argument('-N','--n_clusters',help="Number of clusters to create",default=100,type=int)
    parser.add_argument('-dn','--DatasetName',help="Name of the dataset to analyze", default='filtered_phdata7_condition{}.h5',type=str)
    parser.add_argument('-out','--Out',help="path save",default='/network/lustre/iss02/home/gautam.sridhar/Markov_Fish/Results/',type=str)

    args=parser.parse_args()

    f = h5py.File(args.Out + args.DatasetName + '/kmeans_labels_K{}_N{}_s8646.h5'.format(args.K, args.n_clusters))
    labels_fish = ma.array(f['labels_fish'],dtype=int)
    lengths_all = ma.array(f['MetaData/lengths_data'],dtype=int)
    f.close()

    path_to_filtered_data = '/network/lustre/iss02/home/gautam.sridhar/Markov_Fish/Datasets/JM_Data/'
    P_ensemble = np.load(path_to_filtered_data + 'P_ensemble_ex7_N1200_s8646.npy')

    condition_labels = ['Light (5x5cm)','Light (1x5cm)','Looming(5x5cm)','Dark_Transitions(5x5cm)',
                        'Phototaxis','Optomotor Response (1x5cm)','Optokinetic Response (5x5cm)','Dark (5x5cm)','3 min Light<->Dark(5x5cm)',
                        'Prey Capture Param. (2.5x2.5cm)','Prey Capture Param. RW. (2.5x2.5cm)',
                        'Prey Capture Rot.(2.5x2.5cm)','Prey capture Rot. RW. (2.5x2.5cm)','Light RW. (2.5x2.5cm)']

    condition_recs = np.array([[453,463],[121,133],[49,109],[22,49],[163,193],[109,121],
                               [133,164],[443,453],[0,22],
                               [193,258],[304,387],[258,273],[273,304],
                               [387,443]])

    conditions = np.zeros((np.max(condition_recs),2),dtype='object')
    for k in range(len(condition_recs)):
        t0,tf = condition_recs[k]
        conditions[t0:tf,0] = np.arange(t0,tf)
        conditions[t0:tf,1] = [condition_labels[k] for t in range(t0,tf)]

    to_mask = 1300

    labels_fish[labels_fish == to_mask] = ma.masked

    labels_all= ma.concatenate(labels_fish,axis=0)

    delay = args.Tau

    # lcs,P = op_calc.transition_matrix(labels_all,delay=1,return_connected=True)

    if args.Scale == 2:
        dt = 1
        num_eigfs = 3
        labels_all= ma.concatenate(labels_fish,axis=0)
        P_ensemble = csr_matrix(P_ensemble)
        lcs_ensemble = msm_estimation.largest_connected_set(P_ensemble)
        inv_measure = op_calc.stationary_distribution(P_ensemble)
        final_labels = op_calc.get_connected_labels(labels_all,lcs_ensemble)
        R = op_calc.get_reversible_transition_matrix(P_ensemble)
        eigvals,eigvecs = op_calc.sorted_spectrum(R,k=num_eigfs,seed=43)
        sorted_indices = np.argsort(eigvals.real)[::-1]
        eigvals = eigvals[sorted_indices][1:].real
        eigvals[np.abs(eigvals-1)<1e-12] = np.nan
        eigvals[eigvals<1e-12] = np.nan
        t_imp =  -(delay*dt)/np.log(np.abs(eigvals))
        eigfunctions = eigvecs.real/np.linalg.norm(eigvecs.real,axis=0)

        phi1 = eigfunctions[:,1]
        c_range_phi1,rho_sets,measures,split_idx_phi1,coh_labels_phi1 = op_calc.optimal_partition(phi1,inv_measure,P_ensemble,return_rho=True)
        kmeans_labels = coh_labels_phi1
        cluster_traj_all = ma.copy(final_labels)
        cluster_traj_all[~final_labels.mask] = ma.array(kmeans_labels)[final_labels[~final_labels.mask]]
        cluster_traj_all[final_labels.mask] = ma.masked

        cluster_fish = cluster_traj_all.reshape(labels_fish.shape[0],labels_fish.shape[1])
        cluster_fish_mask = cluster_fish.mask
    else:
        eigfs, inv_measure, final_labels = perform_partition(P_ensemble, labels_fish,delay = args.Tau, n_state = args.Scale)
        cluster_fish, cluster_fish_mask = coarse_grain(eigfs, final_labels, labels_fish, inv_measure, n_state = args.Scale)

    simlabels_fish = to_mask*ma.ones((args.Seeds, len(conditions),np.max(lengths_all)), dtype=int)

    np.random.seed(42)
    n_seeds = np.random.randint(0,10000,args.Seeds)

    params = []
    for s in n_seeds:
        for i in range(labels_fish.shape[0]):
            params.append([s,i,cluster_fish[i]])

    simulate_parallel = functools.partial(simulate_fish, delay)

    # for p in params:
    #     seq = simulate_parallel(p)

    with concurrent.futures.ProcessPoolExecutor() as executor:
       results = executor.map(simulate_parallel, params)

    simlabels = []
    simlengths = []
    simfishes = []
    seeds = []
    for result in results:
        if result is not None:
            s = np.where(n_seeds == result[0])[0][0]
            i = result[1]
            fish_sim = result[2]
            simlabels_fish[s,i, :len(fish_sim)] = fish_sim

    simlabels_fish[simlabels_fish == to_mask] = ma.masked
    print(simlabels_fish[0],flush=True)

    print('Saving results ...',flush=True)
    f = h5py.File(args.Out + args.DatasetName+ '/simlabels_fish_K{}_N{}_tau{}_cg{}.h5'.format(args.K,args.n_clusters,delay,args.Scale),'w')
    metaData = f.create_group('MetaData')
    n_seeds_ =  metaData.create_dataset('n_seeds', len(n_seeds))
    n_seeds_[...] = n_seeds
    lengths_data = metaData.create_dataset('lengths_sims', len(simlengths))
    lengths_data[...] = simlengths
    simfishes_ = metaData.create_dataset('simfishes',len(simfishes))
    simfishes_[...] = simfishes
    simlabels_fish_ = f.create_dataset('sim_labels',simlabels_fish.shape)
    simlabels_fish_[...] = simlabels_fish
    f.close()

    print('Time Taken', time.time() - start_time,flush=True)

if __name__ == "__main__":
    main(sys.argv)
