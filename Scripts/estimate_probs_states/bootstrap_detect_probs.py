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



def main(argv):
    t_0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-len_sim','--L',help="len_sim",default=1000,type=int)
    parser.add_argument('-idx','--Idx',help="idx",default=0,type=int)
    parser.add_argument('-n_bouts','--N',help="n_bouts",default=1000,type=int)
    args=parser.parse_args()
    
    l_idx = args.Idx
    print('l_idx={}'.format(l_idx),flush=True)
    
    l_range = [.2,.4,.6]
    
    len_sim = args.L
    n_bouts = args.N
    
    dr = l_range[l_idx]

    path_to_filtered_data = '/Users/gautam.sridhar/Documents/Repos/Markov_Fish/Datasets/JM_Data/'

    ac_samples_state = []
    for ms in range(4):
        f = h5py.File(path_to_filtered_data+'zebrafish_ms_sims/est_probs/detect_probs_state_{}_nbouts_{}_l_{:.2f}.h5'.format(ms,n_bouts,dr),'r')
        ac_samples = np.array(f['ac_samples'])
        f.close()
        ac_samples_state.append(ac_samples)
    ac_samples_state = np.asarray(ac_samples_state)

    
    import msmtools.estimation as msm_estimation
    from scipy.sparse import csr_matrix
    delay = 3
    dt = 1
    P_ensemble = csr_matrix(np.load(path_to_filtered_data+'/P_ensemble_ex8_N1200_s8684.npy'))
    # lcs_ensemble,P_ensemble = op_calc.transition_matrix(labels_all,delay,return_connected=True)
    lcs_ensemble = msm_estimation.largest_connected_set(P_ensemble)
    inv_measure = op_calc.stationary_distribution(P_ensemble)
    kms2 = np.load(path_to_filtered_data+'/cg4_labels.npy')
    state_measures = [(inv_measure[kms2==state]).sum() for state in np.unique(kms2)]
    print(state_measures)
    
    rmin=0.8
    maxR = 100.
    dist_range = np.logspace(np.log10(rmin),np.log10(maxR),50)
    n_sims = ac_samples_state.shape[1]
    print('n_sims={}'.format(n_sims))
    n_times=1000
    mean = ac_samples_state.mean(axis=1)
    mean_ci = np.zeros((dist_range.shape[0],ac_samples_state.shape[0],3))
    mean_ci_avg = np.zeros((dist_range.shape[0],3))
    for kd in range(dist_range.shape[0]):
        mean_norm = mean[:,kd]/mean[:,kd].sum()
        mean_norm_avg = np.sum([mean_norm[ms]*state_measures[ms] for ms in np.unique(kms2)])
        samples = []
        samples_avg = []
        for k in range(n_times):
            mean_s = ac_samples_state[:,np.random.randint(0,n_sims,n_sims)].mean(axis=1)
            mean_norm_s = mean_s[:,kd]/mean_s[:,kd].sum()
            samples.append(mean_norm_s)
            samples_avg.append(np.sum([mean_norm_s[ms]*state_measures[ms] for ms in np.unique(kms2)]))
        cil = np.percentile(samples,2.5,axis=0)
        ciu = np.percentile(samples,97.5,axis=0)
        cil_avg = np.percentile(samples_avg,2.5)
        ciu_avg = np.percentile(samples_avg,97.5)
        mean_ci[kd,:,0] = mean_norm
        mean_ci[kd,:,1] = cil
        mean_ci[kd,:,2] = ciu
        mean_ci_avg[kd,0] = mean_norm_avg
        mean_ci_avg[kd,1] = cil_avg
        mean_ci_avg[kd,2] = ciu_avg
        print(kd,flush=True)

    
    print(mean_ci.shape,flush=True)
    
    print("Save bootstrapped_probs",flush=True)

    f = h5py.File(path_to_filtered_data+'/zebrafish_ms_sims/est_probs/bootstrap_detect_probs_states_nbouts_{}_l_{:.2f}.h5'.format(n_bouts,dr),'w')
    mc_ = f.create_dataset('mean_ci',mean_ci.shape)
    mc_[...] = mean_ci
    mca_ = f.create_dataset('mean_ci_avg',mean_ci_avg.shape)
    mca_[...] = mean_ci_avg
    dr_ = f.create_dataset('dist_range',dist_range.shape)
    dr_[...] = dist_range
    f.close()
    t_f = time.time()
    print('It took {:.2f} minutes.'.format((t_f-t_0)/60.))
    
if __name__ == "__main__":
    main(sys.argv)
