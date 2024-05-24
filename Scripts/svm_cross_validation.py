import h5py
import argparse
#numpy
import numpy as np
import pandas as pd
import numpy.ma as ma
import functools
import concurrent.futures
# %matplotlib notebook
import sys
sys.path.append('/network/lustre/iss02/home/gautam.sridhar/Markov_Fish/utils/')
import os
import copy
import clustering_methods as cl
import operator_calculations as op_calc
import stats
import time
import scipy
import msmtools.estimation as msm_estimation
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scipy.sparse import diags,identity,coo_matrix, csr_matrix


def CSE(D):
    """
    Perform constant shift embedding and return eigenvalues and eigenvectors on a distance matrix 
    """
    n = D.shape[0]
    
    Q = np.eye(n) - (1/n)*np.dot(np.ones((n,1)), np.ones((1,n)))
    # print(Q)
    Sc = -0.5*(np.dot(Q,np.dot(D,Q)))
    ei1, _ = np.linalg.eig(Sc)
    Dtilde = D - 2*(np.min(ei1))*(np.dot(np.ones((n,1)), np.ones((1,n))) - np.eye(n))

    Sctilde = -0.5*(np.dot(Q,np.dot(Dtilde,Q)))
    ei_fin, ev = np.linalg.eig(Sctilde)
    sorted_idx = np.argsort(ei_fin)[::-1]
    ei_fin = ei_fin[sorted_idx].real
    ev = ev[:,sorted_idx].real
    
    return ei_fin, ev


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
    eigfunctions_traj = ma.array(eigfunctions)[final_labels,:]
    eigfunctions_traj[final_labels.mask] = ma.masked

    split_locs = []
    distorted_eigfs = np.zeros((eigfunctions.shape[0], eigfunctions.shape[1]-1))
    for i in range(1,eigfunctions.shape[1]):
        print(i)
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
    km = KMeans(n_clusters=n_state,n_init=1000,random_state=123).fit(X, sample_weight=inv_measure)
    
    cluster_traj_all = ma.copy(final_labels)
    cluster_traj_all[~final_labels.mask] = ma.array(km.labels_)[final_labels[~final_labels.mask]]
    cluster_traj_all[final_labels.mask] = ma.masked

    cluster_fish = cluster_traj_all.reshape(labels_fish.shape[0],labels_fish.shape[1])
    cluster_fish_mask = cluster_fish.mask

    return cluster_fish, cluster_fish_mask


def calculate_tmats(cluster_fish, cluster_fish_mask, n_state,delay=3):
    """
    Calculates and returns tmats for each fish at the course graining scale of choice
    """
    P_prj = ma.zeros((cluster_fish.shape[0],n_state,n_state))
    lcs_all = []
    for cf in range(cluster_fish.shape[0]):
        lcs_spec,P_spec = op_calc.transition_matrix(cluster_fish[cf],delay=3, return_connected=True)
        lcs_all.append(lcs_spec.astype(int))
        for i,l in enumerate(lcs_spec):
            P_prj[cf, l,lcs_spec] = P_spec.todense()[i]

    return P_prj


def calculate_distances(P_prj):
    """
    Calculate distance matrix between tmats of fish and return the ratio of the nearest to furthest fish
    """

    mean_js = ma.zeros((P_prj.shape[0],P_prj.shape[0]))
    for cf1 in range(P_prj.shape[0]):  
        for cf2 in range(cf1+1):
            mean_js[cf1,cf2] = np.average(np.sum(np.abs(P_prj[cf1] - P_prj[cf2]), axis=1))

    mean_js_full = mean_js + mean_js.T   
    ratio = np.sort(mean_js_full,axis=1)[:,1]/np.sort(mean_js_full,axis=1)[:,-1]
    
    return mean_js_full, ratio


def main(argv):
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('-seeds','--Seeds',help="Number of seeds to evaluate over",type=int,default=10)
    parser.add_argument('-del','--Delay',help="Delay parameter for jumps in markov chain",default=3,type=int)
    parser.add_argument('-n','--N',help="Max number of states to coarse grain into",default=1,type=int)
    parser.add_argument('-dn','--DatasetName',help="Name of the dataset to analyze", default='filtered_phdata7_condition{}.h5',type=str)
    parser.add_argument('-out','--Out',help="path save",default='/network/lustre/iss02/home/gautam.sridhar/Markov_Fish/Results/',type=str)

    args=parser.parse_args()

    condition_labels = ['Light (5x5cm)','Light (1x5cm)','Looming(5x5cm)','ChasingDot coarsespeeds(5x5cm)','ChasingDot finespeeds(5x5cm)','Dark_Transitions(5x5cm)',
                    'Phototaxis','Optomotor Response (1x5cm)','Optokinetic Response (5x5cm)','Dark (5x5cm)','3 min Light<->Dark(5x5cm)',
                    'Prey Capture Param. (2.5x2.5cm)','Prey Capture Param. RW. (2.5x2.5cm)',
                    'Prey Capture Rot.(2.5x2.5cm)','Prey capture Rot. RW. (2.5x2.5cm)','Light RW. (2.5x2.5cm)']


    condition_recs = np.array([[515,525],[160,172],[87,148],[43,60],[22,43],[60,87],
                           [202,232],[148,160],[172,202],[505,515],[0,22],
                           [232,301],[347,445],[301,316],[316,347],
                           [445,505]])

    f = h5py.File(args.Out + args.DatasetName + '/kmeans_labels_K5_N1200_s8684.h5')
    lengths_all = np.array(f['MetaData/lengths_data'], dtype=int)
    labels_fish = ma.array(f['labels_fish'],dtype=int)

    labels_fish[labels_fish == 1300] = ma.masked

    conditions = np.zeros((np.max(condition_recs),2),dtype='object')

    for k in range(len(condition_labels)):
        t0,tf = condition_recs[k]
        conditions[t0:tf,0] = np.arange(t0,tf)
        conditions[t0:tf,1] = [condition_labels[k] for t in range(t0,tf)]

    path_to_filtered_data = '/network/lustre/iss02/home/gautam.sridhar/Markov_Fish/Datasets/JM_Data/'
    recs_remove = np.load(path_to_filtered_data + 'recs_remove.npy')
    recs_remove = np.hstack([recs_remove,np.arange(22,60)])
    P_ensemble = np.load(path_to_filtered_data + 'P_ensemble_ex8_N1200_s8684.npy')

    conditions = np.delete(conditions, recs_remove, axis=0)

    eigfs, inv_measure, final_labels = perform_partition(P_ensemble, labels_fish,delay = args.Delay, n_state = args.N)

    epss = np.logspace(-2,0,15,base=10)
    # alphas = np.logspace()
    # epss = np.array([0.05,0.09,0.1,0.3])

    n_states = [2,3,4,5,6,7,8,9]
    n_states = np.append(n_states, np.round(np.logspace(1,3,25,base=10)).astype(int))


    condition_labels = ['Light (5x5cm)','Light (1x5cm)','Looming(5x5cm)','Dark_Transitions(5x5cm)',
                    'Phototaxis','Optomotor Response (1x5cm)','Optokinetic Response (5x5cm)','Dark (5x5cm)','3 min Light<->Dark(5x5cm)',
                    'Prey Capture Param. (2.5x2.5cm)','Prey Capture Param. RW. (2.5x2.5cm)',
                    'Prey Capture Rot.(2.5x2.5cm)','Prey capture Rot. RW. (2.5x2.5cm)','Light RW. (2.5x2.5cm)']

    grid_values = {'C':np.logspace(-1,1,10,base=10)}

    ratios = []

    conf_matrices = []
    best_alphas = []

    train_acc = []
    test_acc = []

    for n_state in n_states:
        print(n_state,flush=True)
        cluster_fish, cluster_fish_mask = coarse_grain(eigfs, final_labels, labels_fish, inv_measure, n_state)

        P_prj = calculate_tmats(cluster_fish, cluster_fish_mask, n_state, delay=args.Delay)

        mean_js_full, ratio = calculate_distances(P_prj)
        ratios.append(ratio)

        ei_fin, ev = CSE(mean_js_full)
        ei_thres = ei_fin>0.0
        X_input = np.dot(ev[:,ei_thres], np.sqrt(np.diag(ei_fin[ei_thres])))

        y = np.zeros((463,))
        for i,cond in enumerate(condition_labels):
            cond_recs = np.where(conditions[:,1] == cond)[0]
            y[cond_recs] = i

        y_bin = label_binarize(y, classes=np.arange(0,len(condition_labels)))

        np.random.seed(42)
        seeds = np.random.randint(0,100000,100)

        for s in seeds:
            np.random.seed(s)
            train_idx = []
            test_idx = []
            for cond in condition_labels:
                cond_recs = np.where(conditions[:,1] == cond)[0]
                shuffled_idx = np.random.choice(cond_recs, len(cond_recs), replace=False)
                tr_idx = shuffled_idx[:int(0.8*len(shuffled_idx))]
                te_idx = shuffled_idx[int(0.8*len(shuffled_idx)):]
                train_idx.append(tr_idx)
                test_idx.append(te_idx)

            train_idx = np.hstack(train_idx)
            test_idx = np.hstack(test_idx)

            X_train = X_input[train_idx,:]

            X_test = X_input[test_idx,:]

            y_train, y_bintrain = y[train_idx], y_bin[train_idx]
            y_test, y_bintest = y[test_idx], y_bin[test_idx]

            model = GridSearchCV(LogisticRegression(class_weight='balanced', 
                                                    random_state=s, max_iter=1000),param_grid=grid_values)
            model.fit(X_train, y_train)
            Y_out_train = model.predict(X_train)
            Y_out = model.predict(X_test)
            
            class_weights = len(y_train) / (16. * np.bincount(y_train.astype(int)))
            
            sample_weights = np.zeros((y_train.shape[0],))
            for k in np.unique(y_train):
                idx = np.asarray(np.where(y_train == k)[0], dtype=int)
                sample_weights[idx] = class_weights[int(k)]

            print('Train Accuracy', accuracy_score(y_train, Y_out_train, sample_weight = sample_weights),flush=True)
            train_acc.append(accuracy_score(y_train, Y_out_train, sample_weight = sample_weights))
            
            class_weights = len(y_test) / (16. * np.bincount(y_test.astype(int)))
            
            sample_weights = np.zeros((y_test.shape[0],))
            for k in np.unique(y_test):
                idx = np.asarray(np.where(y_test == k)[0], dtype=int)
                sample_weights[idx] = class_weights[int(k)]

            print('Test Accuracy', accuracy_score(y_test, Y_out,sample_weight = sample_weights),flush=True)
            
            test_acc.append(accuracy_score(y_test, Y_out, sample_weight = sample_weights))

            best_alphas.append(model.best_params_['C'])
            conf_matrices.append(confusion_matrix(y_test, Y_out))
    
    train_acc = np.asarray(train_acc)
    test_acc = np.asarray(test_acc)
    best_alphas = np.asarray(best_alphas)
    conf_matrices = np.stack(conf_matrices, axis=0)

    print('Saving results ...',flush=True)
    f = h5py.File(args.Out+args.DatasetName+'/Cross_Val.h5','w')
    train_acc_ = f.create_dataset('train_acc',train_acc.shape)
    train_acc_[...] = train_acc
    test_acc_ = f.create_dataset('test_acc',test_acc.shape)
    test_acc_[...] = test_acc
    best_alphas_ = f.create_dataset('best_alphas',best_alphas.shape)
    best_alphas_[...] = best_alphas
    conf_matrices_ = f.create_dataset('conf_matrices',conf_matrices.shape)
    conf_matrices_[...] = conf_matrices


if __name__ == "__main__":
    main(sys.argv)
