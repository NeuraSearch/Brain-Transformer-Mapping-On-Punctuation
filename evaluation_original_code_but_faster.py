import argparse
import os

import numba
import numpy
import numpy as np
from numba import jit, cuda
import time as tm
import pickle as pk


from brain_language_nlp.utils.utils import CV_ind, binary_classify_neighborhoods

def run_evaluations(input_path,output_path,subject,home_path,method):

    loaded = np.load(input_path, allow_pickle=True)
    preds_t_per_feat = loaded.item()['preds_t']
    test_t_per_feat = loaded.item()['test_t']
    print(test_t_per_feat.shape)

    n_class = 20  # how many predictions to classify at the same time
    n_folds = 4
    neighborhoods = np.load(home_path + '/data/voxel_neighborhoods/' + subject + '_ars_auto2.npy')
    n_words, n_voxels = test_t_per_feat.shape
    ind = CV_ind(n_words, n_folds=n_folds)

    accs = np.zeros([n_folds, n_voxels])
    acc_std = np.zeros([n_folds, n_voxels])

    for ind_num in range(n_folds):
        print("Starting fold "+str(ind_num))
        start_time = tm.time()
        test_ind = ind == ind_num
        acc = np.full([1000, test_t_per_feat[test_ind, :].shape[-1]], np.nan)
        return_acc, _, _, _ = binary_classify_neighborhoods_faster(preds_t_per_feat[test_ind, :],
                                                                  test_t_per_feat[test_ind, :], 20,
                                                                  1000,np.arange(0),
                                                                  np.asarray(neighborhoods,dtype=int),acc)
        accs[ind_num,:] = np.nanmean(return_acc,0)
        print('Classification for fold done. Took {} seconds'.format(tm.time() - start_time))

    fname = output_path 
    if n_class < 20:
        fname = fname + '_{}v{}_'.format(n_class, n_class)

    with open(fname + '_{}_accs.pkl'.format(subject), 'wb') as fout:
        pk.dump(accs, fout)

    print('saved: {}'.format(fname + '_accs.pkl'))
#putthing the @jit annotation tells numba to translate this function to C so it can
#run directly to the hardware
# cache parameter says to the library that if a cached version exists use that.
@jit(nopython=True,cache=True)
def binary_classify_neighborhoods_faster(Ypred, Y, n_class, nSample,pair_samples,neighborhoods,acc):
    # n_class = how many words to classify at once
    # nSample = how many words to classify

    voxels = Y.shape[-1]

    #neighborhoods = np.asarray(neighborhoods,dtype=int64)
    #acc = np.zeros((nSample, Y.shape[-1]))
    #acc2 = np.full([nSample, Y.shape[-1]], np.nan)
    test_word_inds = []

    if pair_samples.size>0:
        Ypred2 = Ypred[pair_samples>=0]
        Y2 = Y[pair_samples>=0]
        pair_samples2 = pair_samples[pair_samples>=0]
    else:
        Ypred2 = Ypred
        Y2 = Y
        pair_samples2 = pair_samples
    n = Y2.shape[0]

    for idx in range(nSample):
        idx_real = np.random.choice(n, n_class)
        sample_real = Y2[idx_real]
        sample_pred_correct = Ypred2[idx_real]

        if pair_samples2.size == 0:
            idx_wrong = np.random.choice(n, n_class)
        else:
           print("Somethgin")
           #idx_wrong = sample_same_but_different(idx_real,pair_samples2)
        sample_pred_incorrect = Ypred2[idx_wrong]

        #print(sample_pred_incorrect.shape)

        # compute distances within neighborhood
        dist_correct = np.sum((sample_real - sample_pred_correct) ** 2, 0)
        dist_incorrect = np.sum((sample_real - sample_pred_incorrect) ** 2, 0)

        neighborhood_dist_correct = get_distances(dist_correct,neighborhoods,voxels)
        neighborhood_dist_incorrect = get_distances(dist_incorrect,neighborhoods,voxels)

        acc[idx,:] = (neighborhood_dist_correct < neighborhood_dist_incorrect)*1.0 + (neighborhood_dist_correct == neighborhood_dist_incorrect)*0.5
        test_word_inds.extend(idx_real)

    return acc, acc, acc, np.array(test_word_inds)
@jit(nopython=True,cache=True)
def get_distances(distances,neighborhoods,voxels):
    neighborhood_distances = []
    for v in range(voxels):
        voxel_neighbhours = neighborhoods[v]
        voxel_neighbhours_new = [value for value in voxel_neighbhours if value != -1]
        sum=0
        for index in voxel_neighbhours_new:
            sum+=distances[index]
        neighborhood_distances.append(sum)
    return np.array(neighborhood_distances)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp_model", default='bert',choices=["bert","roberta","albert","distilibert","electra"])
    parser.add_argument("--sequence_lengths",default="4",help='length of context to provide to NLP model (default: 1)')
    parser.add_argument("--output_dir",default="/media/wrb15144/drives/i/Science/CIS-YASHMOSH/zenonlamprou/neurolinguistics-project/code/fMRI-AI-KB/data/models_output/bert/features/40/",help='directory to save extracted representations to')
    parser.add_argument("--home_path", default=os.getcwd())
    parser.add_argument("--feature_strategy", default="normal",choices=["normal","padding_all","padding_everything","padding_fixations","removing_fixations"])
    parser.add_argument("--method", default="plain",choices=["plain","kernel_ridge","kernel_ridge_svd","svd","ridge_sk"])
    args = parser.parse_args()
    print(args)
    lengths = args.sequence_lengths.split(",")
    for length in lengths:
        starting_point = 0
        for layer in range(starting_point,13):

            f = open(args.home_path+"/data/models_output/{}/evaluations/{}/{}/{}/{}/evaluation_script.sh".format(args.nlp_model,args.feature_strategy,args.method,length,layer), "r")
            lines = f.readlines()
            f.close()
            for line in lines:
                line = line.replace("gnome-terminal -- ","")
                tokens = line.split(" ")
                subject = tokens[4]
                method = tokens[6]
                input_path = tokens[8]
                output_path = tokens[-1].replace("\n","")
                print("Running")
                print(input_path)
                print(output_path)
                print(subject)

                run_evaluations(input_path,output_path,subject,args.home_path,method)


