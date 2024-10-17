#data format library
import argparse
import copy
import os
import sys
import functools
import concurrent.futures
sys.path.append('/Users/gautam.sridhar/Documents/Repos/Markov_Fish/utils/') ## Update Path here
import h5py
#numpy
import numpy as np
import pandas as pd
import numpy.ma as ma
import clustering_methods as cl
import delay_embedding as embed
import operator_calculations as op_calc
import stats
import matplotlib.pyplot as plt

import time

np.random.seed(42)


def calc_entropy_from_data(data, bouts, conditions,params):
    """
    Calculate the  for a given seed, delay K and cluster size.
    Parameters:
        data: datasets to perform the calculations on
        bouts: number of bouts to sample from each condition
        params: list containing seed, delay K and number of cluster N_cluster
    Returns:
            list with value of seed, delay, number of cluster and entropies
    """

    seed = params[0]
    K = params[1]
    N_cluster = params[2]

    min_count = 200
    data_bootstrap = np.asarray(copy.deepcopy(data),dtype='object')
    print('Seed:{}'.format(seed))
    np.random.seed(seed)

    condition_labels = ['Light (5x5cm)','Light (1x5cm)','Looming(5x5cm)','Dark_Transitions(5x5cm)',
                        'Phototaxis','Optomotor Response (1x5cm)','Optokinetic Response (5x5cm)','Dark (5x5cm)','3 min Light<->Dark(5x5cm)',
                        'Prey Capture Param. (2.5x2.5cm)','Prey Capture Param. RW. (2.5x2.5cm)',
                        'Prey Capture Rot.(2.5x2.5cm)','Prey capture Rot. RW. (2.5x2.5cm)','Light RW. (2.5x2.5cm)']

    sampled_data = []
    for cond in condition_labels:
        cond_recs = np.where(conditions[:,1] == cond)[0]
        samples_ = ma.zeros((bouts+4, data_bootstrap[0].shape[1]))
        data_cond = ma.concatenate(data_bootstrap[cond_recs], axis=0)
        start_pos = np.random.randint(0, len(data_cond)-bouts)
        samples_[2:-2] = data_cond[start_pos:start_pos+bouts]
        samples_[samples_ == 0] = ma.masked
        sampled_data.append(samples_)
 
    H = []
    #for kf,f0 in enumerate(np.arange(0,data_bootstrap.shape[0],div)):
    pc_ = ma.concatenate(sampled_data,axis=0)
    print('Starting delay {}, clusters {}:'.format(K, N_cluster), pc_.shape, flush=True)
    if ma.count(pc_,axis=0)[0]>min_count:
        traj_matrix = embed.trajectory_matrix(pc_,K=K-1)
        labels = cl.kmeans_knn_partition(traj_matrix,N_cluster,batchsize=5000)
        h = op_calc.get_entropy(labels)
        H.append(h)
    print('Finished processing: delay {}, clusters {}'.format(K,N_cluster), traj_matrix.shape,flush=True)

    return [seed, K, N_cluster, H]


def main(argv):
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('-seeds','--Seeds',help="Number of seeds to evaluate over",type=int,default=10)
    parser.add_argument('-dn','--DatasetName',help="Name of the dataset to analyze", default='filtered_phdata7_condition{}.h5',type=str)
    parser.add_argument('-bouts','--Bouts',help="Number of bouts to sample from each condition",default=1000,type=int)
    parser.add_argument('-out','--Out',help="path save",default='/network/lustre/iss02/home/gautam.sridhar/Markov_Fish/Results/',type=str)

    args=parser.parse_args()

    path_to_filtered_data = '/network/lustre/iss02/home/gautam.sridhar/Markov_Fish/Datasets/JM_Data/'
    recs_remove = np.load(path_to_filtered_data + 'recs_remove.npy')
    recs_remove = np.hstack([recs_remove,np.arange(22,60)])

    f = h5py.File(path_to_filtered_data+args.DatasetName + '.h5','r')

    pca_fish = ma.array(f['pca_fish'])[:,:,:20]
    pca_fish[pca_fish==0] = ma.masked
    print(pca_fish.shape, flush=True)
    lengths = ma.array(f['MetaData/lengths_data'],dtype=int)[:]

    record_list = []
    for i,l in enumerate(lengths):
        pca_fish[i,l-1,:] = ma.masked
        record_list.append(pca_fish[i,:l,:])

    f.close()

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

    min_count=200
    n_seeds = np.random.randint(0,10000,size=args.Seeds) # number of seeds for checking randomness
    K_range = np.arange(1,10,1) #range of delays
    n_clusters=np.arange(50,2250,100) #number of partitions to explore

    # bootstrap_range = (pca_fish.shape[0]//args.Div)*args.Div
    # divs = np.arange(0,bootstrap_range,args.Div)

    params = []
    for s in n_seeds:
        for k in K_range:
            for cs in n_clusters:
                params.append([s,k,cs])
    #print(params)

    h_K = ma.zeros((len(n_seeds),len(K_range),len(n_clusters)))
    calc_ent = functools.partial(calc_entropy_from_data, record_list, args.Bouts, conditions)

    results = []
    # for i in range(len(params)):
    #   results.append(calc_ent(params[i]))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(calc_ent, params)

    for result in results:

         s = np.where(n_seeds == result[0])[0][0]
         K = np.where(K_range == result[1])[0][0]
         cs = np.where(n_clusters == result[2])[0][0]
         # f0s = np.where(divs == result[3])[0][0]
         H = result[3]
         h_K[s, K, cs] = np.asarray(H)[:]

    h_K = ma.array(h_K)
    h_K[h_K==0] = ma.masked

    print('done', flush=True)

    for s,seed in enumerate(n_seeds):
        
        save_path = args.Out + args.DatasetName + '/seed_{}/'.format(seed)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fig, ax = plt.subplots(1,1,figsize=(10,10))
        colors_ = plt.cm.viridis(np.linspace(0,1,len(K_range)))
        for k,K in enumerate(K_range):
            ax.plot(n_clusters,h_K[s,k,:].squeeze(), marker='o', label='delay_{}'.format(K), color=colors_[k])
        #    ax.xticks(fontsize=12)
        #    ax.yticks(fontsize=12)
            ax.set_xlabel(r'$N$ (Number of clusters)',fontsize=14)
            ax.set_ylabel(r'$h$ (nats/bout)',fontsize=14)
            ax.set_title('Entopies for differnt cluster sizes')
            fig.savefig(save_path + 'Entropy_clusters.png')
        ax.legend()
        
        plt.close()

        fig, ax = plt.subplots(2,1,figsize=(10,10))
        colors_ = plt.cm.Reds(np.linspace(0,1,len(n_clusters)))
        for ks,cs in enumerate(n_clusters):
            ax[0].plot(K_range, h_K[s,:,ks].squeeze(), marker='o', label='cluster_{}'.format(cs), color=colors_[ks])
            #ax[0].set_xticks(fontsize=12)
            #ax[0].set_yticks(fontsize=12)
            ax[0].set_xlabel('$K$ (bouts)',fontsize=14)
            ax[0].set_ylabel('$h$ (nats/bout)',fontsize=14)
            ax[0].set_title('Entropy of state')

            ax[0].plot(K_range[:-1], -np.diff(h_K[s,:,ks].squeeze()), marker='o', label='cluster_{}'.format(cs), color=colors_[ks])
            #ax[1].set_xticks(fontsize=12)
            #ax[1].set_yticks(fontsize=12)
            ax[1].set_xlabel('$K$ (bouts)',fontsize=14)
            ax[1].set_ylabel(r'$\partial h/\partial K$ ($nats/bout^2$)',fontsize=14)
            ax[1].set_title('Rate of Entropy')
            ax[1].axhline(0,ls='--',c='k')
        ax[0].legend()
        ax[1].legend()
        fig.savefig(save_path + 'entropy_rate.png')
        plt.close()


    print('Saving results ...', flush=True)
    f = h5py.File(args.Out+ args.DatasetName + '/Entropy_seeds_delays_clusters.h5','w')
    entropies_ = f.create_dataset('entropies',h_K.shape)
    entropies_[...] = h_K
    n_seeds_ = f.create_dataset('seeds',n_seeds.shape)
    n_seeds_[...] = n_seeds
    K_range_ = f.create_dataset('K_range',K_range.shape)
    K_range_[...] = K_range
    n_clusters_ = f.create_dataset('n_clusters',n_clusters.shape)
    n_clusters_[...] = n_clusters


    print('Ended run, total time taken was', time.time() - start_time, flush=True)


if __name__ == "__main__":
    main(sys.argv)
