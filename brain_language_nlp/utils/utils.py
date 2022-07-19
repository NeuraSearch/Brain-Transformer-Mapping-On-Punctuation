import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore
import time
import csv
import os
import nibabel
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage.filters import gaussian_filter
from numpy.linalg import inv, svd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
import time
from scipy.stats import zscore

import time as tm




def load_transpose_zscore(file):
    dat = nibabel.load(file).get_data()
    dat = dat.T
    return zscore(dat,axis = 0)

def smooth_run_not_masked(data,smooth_factor):
    smoothed_data = np.zeros_like(data)
    for i,d in enumerate(data):
        smoothed_data[i] = gaussian_filter(data[i], sigma=smooth_factor, order=0, output=None,
                 mode='reflect', cval=0.0, truncate=4.0)
    return smoothed_data

def delay_one(mat, d):
        # delays a matrix by a delay d. Positive d ==> row t has row t-d
    new_mat = np.zeros_like(mat)
    if d>0:
        new_mat[d:] = mat[:-d]
    elif d<0:
        new_mat[:d] = mat[-d:]
    else:
        new_mat = mat
    return new_mat

def delay_mat(mat, delays):
        # delays a matrix by a set of delays d.
        # a row t in the returned matrix has the concatenated:
        # row(t-delays[0],t-delays[1]...t-delays[last] )
    new_mat = np.concatenate([delay_one(mat, d) for d in delays],axis = -1)
    return new_mat

# train/test is the full NLP feature
# train/test_pca is the NLP feature reduced to 10 dimensions via PCA that has been fit on the training data
# feat_dir is the directory where the NLP features are stored
# train_indicator is an array of 0s and 1s indicating whether the word at this index is in the training set
def get_nlp_features_fixed_length(layer, seq_len, feat_type, feat_dir, train_indicator, SKIP_WORDS=20, END_WORDS=5176):

    loaded = np.load(feat_dir + feat_type + '_length_'+str(seq_len)+ '_layer_' + str(layer) + '.npy')
    if feat_type == 'elmo':
        train = loaded[SKIP_WORDS:END_WORDS,:][:,:512][train_indicator]   # only forward LSTM
        test = loaded[SKIP_WORDS:END_WORDS,:][:,:512][~train_indicator]   # only forward LSTM
    elif feat_type == 'bert' or feat_type == 'transformer_xl' or feat_type == 'use' or feat_type=="distilibert" or feat_type=="albert" or feat_type=="roberta" or feat_type=="electra":
        train = loaded[SKIP_WORDS:END_WORDS,:][train_indicator]
        test = loaded[SKIP_WORDS:END_WORDS,:][~train_indicator]
    else:
        print('Unrecognized NLP feature type {}. Available options elmo, bert, transformer_xl, use'.format(feat_type))

    pca = PCA(n_components=10, svd_solver='full')
    pca.fit(train)
    train_pca = pca.transform(train)
    test_pca = pca.transform(test)

    return train, test, train_pca, test_pca

def CV_ind(n, n_folds):
    ind = np.zeros((n))
    n_items = int(np.floor(n/n_folds))
    for i in range(0,n_folds -1):
        ind[i*n_items:(i+1)*n_items] = i
    ind[(n_folds-1)*n_items:] = (n_folds-1)
    return ind

def TR_to_word_CV_ind(TR_train_indicator,home_path,SKIP_WORDS=20,END_WORDS=5176):
    time = np.load(home_path+'/data/fMRI/time_fmri.npy')
    runs = np.load(home_path+'/data/fMRI/runs_fmri.npy')
    time_words = np.load(home_path+'/data/fMRI/time_words_fmri.npy')
    time_words = time_words[SKIP_WORDS:END_WORDS]

    word_train_indicator = np.zeros([len(time_words)], dtype=bool)
    words_id = np.zeros([len(time_words)],dtype=int)
    # w=find what TR each word belongs to
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i]> time)[0][-1]

        if words_id[i] <= len(runs) - 15:
            offset = runs[int(words_id[i])]*20 + (runs[int(words_id[i])]-1)*15
            if TR_train_indicator[int(words_id[i])-offset-1] == 1:
                word_train_indicator[i] = True

    return word_train_indicator


def prepare_fmri_features(train_features, test_features, word_train_indicator, TR_train_indicator,home_path, SKIP_WORDS=20, END_WORDS=5176):
    time = np.load(home_path+'/data/fMRI/time_fmri.npy')
    runs = np.load(home_path+'/data/fMRI/runs_fmri.npy')
    time_words = np.load(home_path+'/data/fMRI/time_words_fmri.npy')
    time_words = time_words[SKIP_WORDS:END_WORDS]

    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i]> time)[0][-1]
    all_features = np.zeros([time_words.shape[0], train_features.shape[1]])
    all_features[word_train_indicator] = train_features
    all_features[~word_train_indicator] = test_features

    p = all_features.shape[1]
    tmp = np.zeros([time.shape[0], p])
    #The words presented to the participants are first grouped by the TR interval in which they were presented
    for i in range(time.shape[0]):
        # the features from layer l of the words in every group are averaged to form a sequence of features which are aligned to brain signals
        #(words_id<=i)*(words_id>i-1) means which words_id are
        tmp[i] = np.mean(all_features[(words_id<=i)*(words_id>i-1)],0)
    #the models are trained to predict the signal at time t using the concatenated vector formed
    #the delay_mat functions creates that concatinated vector
    tmp = delay_mat(tmp, np.arange(1,5))
    # remove the edges of each run
    tmp = np.vstack([zscore(tmp[runs==i][20:-15]) for i in range(1,5)])
    tmp = np.nan_to_num(tmp)
    return tmp[TR_train_indicator], tmp[~TR_train_indicator]



def run_class_time_CV_fmri_crossval_ridge(data, predict_feat_dict,home_path,method,
                                          regress_feat_names_list = [],
                                          lambdas = np.array([0.1,1,10,100,1000]),
                                          detrend = False, n_folds = 4, skip=5):

    nlp_feat_type = predict_feat_dict['nlp_feat_type']
    feat_dir = predict_feat_dict['nlp_feat_dir']
    layer = predict_feat_dict['layer']
    seq_len = predict_feat_dict['seq_len']


    n_words = data.shape[0]
    n_voxels = data.shape[1]

    ind = CV_ind(n_words, n_folds=n_folds)
    corrs = np.zeros((n_folds, n_voxels))
    acc = np.zeros((n_folds, n_voxels))
    acc_std = np.zeros((n_folds, n_voxels))

    all_test_data = []
    all_preds = []


    for ind_num in range(n_folds):
        train_ind = ind!=ind_num
        test_ind = ind==ind_num

        word_CV_ind = TR_to_word_CV_ind(train_ind,home_path)
        #load the features extracted from BERT or whatever model and use PCA to reduce their dimension to 10
        _,_,tmp_train_features,tmp_test_features = get_nlp_features_fixed_length(layer, seq_len, nlp_feat_type, feat_dir, word_CV_ind)

        #so the train_features and test_features are:
        #features derived from bert from a particular layer
        #align with the time stamps and averaged and then group together
        #for every TR you find which word are in the TR : their index
        # so if word 5 is in a TR then sequence 5 is in the TR
        # because all the features are averaged across that particular word

        train_features,test_features = prepare_fmri_features(tmp_train_features, tmp_test_features, word_CV_ind, train_ind,home_path)
        # split data
        train_data = data[train_ind]
        test_data = data[test_ind]

        # skip TRs between train and test data
        if ind_num == 0: # just remove from front end
            train_data = train_data[skip:,:]
            train_features = train_features[skip:,:]
        elif ind_num == n_folds-1: # just remove from back end
            train_data = train_data[:-skip,:]
            train_features = train_features[:-skip,:]
        else:
            test_data = test_data[skip:-skip,:]
            test_features = test_features[skip:-skip,:]

        # normalize data
        train_data = np.nan_to_num(zscore(np.nan_to_num(train_data)))
        test_data = np.nan_to_num(zscore(np.nan_to_num(test_data)))
        all_test_data.append(test_data)

        train_features = np.nan_to_num(zscore(train_features))
        test_features = np.nan_to_num(zscore(test_features))

        start_time = tm.time()
        #train features are the extracted features from bert
        #train_data are the recordings for each subject
        weights, chosen_lambdas = cross_val_ridge(train_features,train_data, n_splits = 10, lambdas = np.array([10**i for i in range(-6,10)]), method = method,do_plot = False)

        preds = np.dot(test_features, weights)
        corrs[ind_num,:] = corr(preds,test_data)
        all_preds.append(preds)

        print('fold {} completed, took {} seconds'.format(ind_num, tm.time()-start_time))
        del weights

    return corrs, acc, acc_std, np.vstack(all_preds), np.vstack(all_test_data)

def binary_classify_neighborhoods(Ypred, Y, n_class=20, nSample = 1000,pair_samples = [],neighborhoods=[]):
    # n_class = how many words to classify at once
    # nSample = how many words to classify

    voxels = Y.shape[-1]
    neighborhoods = np.asarray(neighborhoods, dtype=int)

    import time as tm

    acc = np.full([nSample, Y.shape[-1]], np.nan)
    acc2 = np.full([nSample, Y.shape[-1]], np.nan)
    test_word_inds = []

    if len(pair_samples)>0:
        Ypred2 = Ypred[pair_samples>=0]
        Y2 = Y[pair_samples>=0]
        pair_samples2 = pair_samples[pair_samples>=0]
    else:
        Ypred2 = Ypred
        Y2 = Y
        pair_samples2 = pair_samples
    n = Y2.shape[0]
    start_time = tm.time()
    for idx in range(nSample):

        idx_real = np.random.choice(n, n_class)
        sample_real = Y2[idx_real]
        sample_pred_correct = Ypred2[idx_real]

        if len(pair_samples2) == 0:
            idx_wrong = np.random.choice(n, n_class)
        else:
            idx_wrong = sample_same_but_different(idx_real,pair_samples2)
        sample_pred_incorrect = Ypred2[idx_wrong]

        #print(sample_pred_incorrect.shape)

        # compute distances within neighborhood
        dist_correct = np.sum((sample_real - sample_pred_correct)**2,0)
        dist_incorrect = np.sum((sample_real - sample_pred_incorrect)**2,0)

        neighborhood_dist_correct = np.array([np.sum(dist_correct[neighborhoods[v,neighborhoods[v,:]>-1]]) for v in range(voxels)])

        neighborhood_dist_incorrect = np.array([np.sum(dist_incorrect[neighborhoods[v,neighborhoods[v,:]>-1]]) for v in range(voxels)])


        acc[idx,:] = (neighborhood_dist_correct < neighborhood_dist_incorrect)*1.0 + (neighborhood_dist_correct == neighborhood_dist_incorrect)*0.5
        test_word_inds.append(idx_real)
    print('Classification for fold done. Took {} seconds'.format(tm.time()-start_time))
    return np.nanmean(acc,0), np.nanstd(acc,0), acc, np.array(test_word_inds)
def corr(X,Y):
    return np.mean(zscore(X)*zscore(Y),0)

def R2(Pred,Real):
    SSres = np.mean((Real-Pred)**2,0)
    SStot = np.var(Real,0)
    return np.nan_to_num(1-SSres/SStot)

def R2r(Pred,Real):
    R2rs = R2(Pred,Real)
    ind_neg = R2rs<0
    R2rs = np.abs(R2rs)
    R2rs = np.sqrt(R2rs)
    R2rs[ind_neg] *= - 1
    return R2rs

#functions on getting the weights
def ridge(X,Y,lmbda):
    # so i think the x axis is the timestamp features from bert and y is the fMRI features
    # np eye : Return a 2-D array with ones on the diagonal and zeros elsewhere. probably this is for the slope
    # T is making it possible for X and Y to be used in the dot method because dot methods does matrix multiplication and
    # in order for that to happen the rows of one matrix needs to match the number of columns of the other matrix
    # X.T.dot(X) sum of square residuals . you do the transpose so you can use the dot product on itself
    # lmbda*np.eye(X.shape[1]) then you add to the square residuals the lambda value times the slope^2 which is a matrix of 40x40 with ones and zeros
    # then you invert the whole thing back so you can do a dot product with the other matrix
    #the other matrix is the dot product of extracted features and fMRI data
    #X.T is used so it can be multipled with Y
    #so essentially the outer dot product is giving a product of matrix multiplication of the ridge regression model and the product of extracted features and fMRI features
    #this is probably the weights of the model
    return np.dot(inv(X.T.dot(X)+lmbda*np.eye(X.shape[1])),X.T.dot(Y))
def ridge_sk(X,Y,lmbda):
    #same thing as ridge function but using sklearn library instead doing it manually
    rd = Ridge(alpha = lmbda)
    '''
    print(X.shape)
    print(Y.shape)
    print(lmbda)
    if(Y.shape[1]==0):
        print(Y)
    print("===========")
    '''
    rd.fit(X,Y)
    return rd.coef_.T

def ridgeCV_sk(X,Y,lmbdas):
    rd = RidgeCV(alphas = lmbdas)
    rd.fit(X,Y)
    return rd.coef_.T

def ridge_svd(X,Y,lmbda):
    U, s, Vt = svd(X, full_matrices=False)
    d = s / (s** 2 + lmbda)
    return np.dot(Vt,np.diag(d).dot(U.T.dot(Y)))

def kernel_ridge(X,Y,lmbda):
    #so the kernel ridge regression is defined as
    #xTranspose times X times inverted(XTranspose times X + lambda times identity matrix) times Y
    # X.T is the first transpose XTranspose
    #inv is the inverted product
    #X.dot(X.T) is the X times X transpose
    #np.eye is the identity matrix
    # lmbda * np.eye  lambda times identity matrix
    # so the whole thing gives us the ridge regression with kernel
    return np.dot(X.T.dot(inv(X.dot(X.T)+lmbda*np.eye(X.shape[0]))),Y)

def kernel_ridge_svd(X,Y,lmbda):
    U, s, Vt = svd(X.T, full_matrices=False)
    d = s / (s** 2 + lmbda)
    return np.dot(np.dot(U,np.diag(d).dot(Vt)),Y)

#functions that get the wheights and output an error score for each lambda for each set of weights

def ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    #x,y,xval,yval are the same thing just split into training and testing
    #testing is to calculate the errors
    #
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    # for every lambda calculate an error
    # lambdas are an array of values
    for idx,lmbda in enumerate(lambdas):
        weights = ridge(X,Y,lmbda)
        #error for every lambda is calculate
        #1-R2
        #get back an error for every lambda
        #np.dot(Xval,weights) these are the predictions. Essentialy is the question if we combine the model weights with the extracted features from testing can we predict
        #accuretly the fMRI recordings
        # Yval these are the labels
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error

def ridge_by_lambda_sk(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))

    for idx,lmbda in enumerate(lambdas):
        weights = ridge_sk(X,Y,lmbda)
        error[idx] = 1 -  R2(np.dot(Xval,weights),Yval)
    return error



def ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    # svd stands for singular value decomposition
    # decompose matrix X into the product of 3 different matrixes = U ,S,V(transpose)
    # so if a has shape MxN then  U = MxM S=MxN and Vt=NxN
    # U and Vt are unitary matrices. if we multiply one of these matrices by its transpose (or the other way around), the result equals the identity matrix.
    # On the other hand, the matrix S
    # is diagonal, and it stores non-negative singular values ordered by relevance.
    U, s, Vt = svd(X, full_matrices=False)

    for idx,lmbda in enumerate(lambdas):
        # S matrix is diagonal, only the first n row diagonal values are worth keeping.
        #s columns are order in descending order
        #they show how important each column is for U and Vt

        # Indeed the last n rows of S are filled with 0s. For this reason, it is very common to keep only the first r×r non-negative diagonal values of S
        # we can cut off some rows from s for being 0 or close to 0 because that means they are not that importan
        #I am assuming the line below does that. Cutting off the lines based on the slop(identity matrix) and lambda
        #because in ridge regression the penalty is applied by slope^2+lambda
        #so cutting the rows by the penalty
        #that gives a new identity matrix
        d = s / (s** 2 + lmbda)

        # d = s divided by s^2 + lambda
        # this equation will be a dot product of
        # 1. Vt
        # 2. second input is a dot product of
        #   the diagonal of d and U transpose times Y
        # np.diag Extract a diagonal or construct a diagonal array.
        # so basically because d is just a list in order to use it in matrix multiplication
        #np.diag constructs a matrix of 40x40 with everything zero except the diagonal.
        #d is the identity matrix
        #Vt and U represents the important columns based on penalty
        #so bellow is the ridge regression of how far the timestamped bert features are (Vt) using the regression model with the Identity matrix
        #times the timestamped data and the fMRI data.-`
        weights = np.dot(Vt,np.diag(d).dot(U.T.dot(Y)))
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error

def kernel_ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    #using X transpose instead of X
    #getting X^2 for the kernel
    U, s, Vt = svd(X.T, full_matrices=False)
    for idx,lmbda in enumerate(lambdas):
        #creating the new identity matrix using the ridge regression penalty to cut off the unimportant rows.
        d = s / (s** 2 + lmbda)
        #se how far of is the timestamped(U,Vt) bert features from the fMRI recordings(Y)
        #np.dot(U,np.diag(d).dot(Vt)) this is the ridge regression model
        #combining the important columns based on the new identity matrix
        weights = np.dot(np.dot(U,np.diag(d).dot(Vt)),Y)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error

def kernel_ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = kernel_ridge(X,Y,lmbda)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error



def kernel_ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    U, s, Vt = svd(X.T, full_matrices=False)
    for idx,lmbda in enumerate(lambdas):
        d = s / (s** 2 + lmbda)
        weights = np.dot(np.dot(U,np.diag(d).dot(Vt)),Y)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error


#main loop that utilises the above functions
def cross_val_ridge(train_features,train_data, n_splits = 10,
                    lambdas = np.array([10**i for i in range(-6,10)]),
                    method = 'plain',
                    do_plot = False):

    ridge_1 = dict(plain = ridge_by_lambda,
                   svd = ridge_by_lambda_svd,
                   kernel_ridge = kernel_ridge_by_lambda,
                   kernel_ridge_svd = kernel_ridge_by_lambda_svd,
                   ridge_sk = ridge_by_lambda_sk)[method]
    ridge_2 = dict(plain = ridge,
                   svd = ridge_svd,
                   kernel_ridge = kernel_ridge,
                   kernel_ridge_svd = kernel_ridge_svd,
                   ridge_sk = ridge_sk)[method]

    n_voxels = train_data.shape[1]
    nL = lambdas.shape[0]
    r_cv = np.zeros((nL, train_data.shape[1]))

    kf = KFold(n_splits=n_splits)
    start_t = time.time()
    for icv, (trn, val) in enumerate(kf.split(train_data)):
        #print('ntrain = {}'.format(train_features[trn].shape[0]))
        cost = ridge_1(train_features[trn],train_data[trn],
                               train_features[val],train_data[val],
                               lambdas=lambdas)
        if do_plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(cost,aspect = 'auto')
        #get the cost for every lambda value
        r_cv += cost
        #if icv%3 ==0:
        #    print(icv)
        #print('average iteration length {}'.format((time.time()-start_t)/(icv+1)))
    if do_plot:
        plt.figure()
        plt.imshow(r_cv,aspect='auto',cmap = 'RdBu_r');
    #get the best run index (lambda) for every voxel
    argmin_lambda = np.argmin(r_cv,axis = 0)
    weights = np.zeros((train_features.shape[1],train_data.shape[1]))
    for idx_lambda in range(lambdas.shape[0]): # this is much faster than iterating over voxels!
        idx_vox = argmin_lambda == idx_lambda
        #get the best possible prediction for a different lambda
        if train_data[:,idx_vox]!=[]:
            weights[:,idx_vox] = ridge_2(train_features, train_data[:,idx_vox],lambdas[idx_lambda])
    if do_plot:
        plt.figure()
        plt.imshow(weights,aspect='auto',cmap = 'RdBu_r',vmin = -0.5,vmax = 0.5);
    return weights, np.array([lambdas[i] for i in argmin_lambda])