import json

from utils.ELECTRA_Utils import get_electra_layer_representations
from utils.ALBERT_Utils import get_albert_layer_representations
from utils.DistilBERT_utils import get_distiliBert_layer_representations
from utils.RoBERTa_Utils import get_roberta_layer_representations
from utils.bert_utils import get_bert_layer_representations
from utils.xl_utils import get_xl_layer_representations
#from utils.elmo_utils import get_elmo_layer_representations
from utils.use_utils import get_use_layer_representations

import time as tm
import numpy as np
import torch
import os
import argparse

                
def save_layer_representations(model_layer_dict, model_name, seq_len, save_dir):
    for layer in model_layer_dict.keys():
        np.save('{}/{}_length_{}_layer_{}.npy'.format(save_dir,model_name,seq_len,layer+1),np.vstack(model_layer_dict[layer]))  
    print('Saved extracted features to {}'.format(save_dir))
    return 1

                
model_options = ['bert','transformer_xl','elmo','use',"albert","roberta","distilibert","electra"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp_model", default='bert', choices=model_options)
    parser.add_argument("--sequence_length", type=int, default=40, help='length of context to provide to NLP model (default: 1)')
    parser.add_argument("--output_dir",default="/media/wrb15144/drives/i/Science/CIS-YASHMOSH/zenonlamprou/neurolinguistics-project/code/fMRI-AI-KB/data/models_output/bert/features/40/", help='directory to save extracted representations to')
    parser.add_argument("--home_path", default="/home/wrb15144/zenon/fMRI-AI-KB")
    parser.add_argument("--feature_strategy", default="normal")
    args = parser.parse_args()
    print(args)
    text_array = np.load(args.home_path+'/data/stimuli_words.npy')
    remove_chars = [",","\"","@"]
    
    
    if args.nlp_model == 'bert':
        # the index of the word for which to extract the representations (in the input "[CLS] word_1 ... word_n [SEP]")
        # for CLS, set to 0; for SEP set to -1; for last word set to -2
        word_ind_to_extract = -2
        nlp_features = get_bert_layer_representations(args.sequence_length, text_array, remove_chars, word_ind_to_extract,args.feature_strategy,args.home_path)
    elif args.nlp_model == 'albert':
        # the index of the word for which to extract the representations (in the input "[CLS] word_1 ... word_n [SEP]")
        # for CLS, set to 0; for SEP set to -1; for last word set to -2
        word_ind_to_extract = -2
        nlp_features = get_albert_layer_representations(args.sequence_length, text_array, remove_chars,
                                                      word_ind_to_extract)
    elif args.nlp_model == 'electra':
        # the index of the word for which to extract the representations (in the input "[CLS] word_1 ... word_n [SEP]")
        # for CLS, set to 0; for SEP set to -1; for last word set to -2
        word_ind_to_extract = -2
        nlp_features = get_electra_layer_representations(args.sequence_length, text_array, remove_chars,
                                                        word_ind_to_extract)

    elif args.nlp_model == 'distilibert':
        # the index of the word for which to extract the representations (in the input "[CLS] word_1 ... word_n [SEP]")
        # for CLS, set to 0; for SEP set to -1; for last word set to -2
        word_ind_to_extract = -2
        nlp_features = get_distiliBert_layer_representations(args.sequence_length, text_array, remove_chars,
                                                      word_ind_to_extract)

    elif args.nlp_model == 'roberta':
        # the index of the word for which to extract the representations (in the input "[CLS] word_1 ... word_n [SEP]")
        # for CLS, set to 0; for SEP set to -1; for last word set to -2
        word_ind_to_extract = -2
        nlp_features = get_roberta_layer_representations(args.sequence_length, text_array, remove_chars,
                                                             word_ind_to_extract,args.feature_strategy,args.home_path)
    elif args.nlp_model == 'transformer_xl':
        word_ind_to_extract = -1
        nlp_features = get_xl_layer_representations(args.sequence_length, text_array, remove_chars, word_ind_to_extract)
    elif args.nlp_model == 'elmo':
        word_ind_to_extract = -1
        #nlp_features = get_elmo_layer_representations(args.sequence_length, text_array, remove_chars, word_ind_to_extract)
    elif args.nlp_model == 'use':
        nlp_features = get_use_layer_representations(args.sequence_length, text_array, remove_chars)
    else:
        print('Unrecognized model name {}'.format(args.nlp_model))
        
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)          
    #create save_directory

    save_layer_representations(nlp_features, args.nlp_model, args.sequence_length, args.output_dir)
        
        
        
        
    
    
    

    
