import argparse
import os
from pathlib import Path

import numpy as np

from utils.utils import run_class_time_CV_fmri_crossval_ridge
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--nlp_feat_type", required=True)
    parser.add_argument("--nlp_feat_dir", required=True)
    parser.add_argument("--layer", type=int, required=False)
    parser.add_argument("--sequence_length", type=int, required=False)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--home_path", default="/media/wrb15144/drives/i/Science/CIS-YASHMOSH/zenonlamprou/neurolinguistics-project/code/fMRI-AI-KB")
    parser.add_argument("--method",default="plain")
    args = parser.parse_args()
    print(args)
        
    predict_feat_dict = {'nlp_feat_type':args.nlp_feat_type,
                         'nlp_feat_dir':args.nlp_feat_dir,
                         'layer':args.layer,
                         'seq_len':args.sequence_length}


    # loading fMRI data
    print(args.home_path)
    data = np.load(args.home_path+'/data/fMRI/data_subject_{}.npy'.format(args.subject))
    corrs_t, _, _, preds_t, test_t = run_class_time_CV_fmri_crossval_ridge(data,predict_feat_dict,args.home_path,args.method)

    overall_output_dir = args.output_dir+"/"+args.method+"/"
    if not os.path.exists(overall_output_dir):
        os.mkdir(overall_output_dir)
    fname = 'predict_{}_with_{}_layer_{}_len_{}'.format(args.subject, args.nlp_feat_type, args.layer, args.sequence_length)

    print('saving: {}'.format(overall_output_dir + fname))
    if not os.path.exists(overall_output_dir):
        Path(overall_output_dir).mkdir(parents=True, exist_ok=True)
    np.save(overall_output_dir + fname + '.npy', {'corrs_t':corrs_t,'preds_t':preds_t,'test_t':test_t})

    
