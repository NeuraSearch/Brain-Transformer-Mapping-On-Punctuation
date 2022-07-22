import argparse
import os

def construct_commands(f,features_path,out_path,sequence_length,feat_type,layers,method):
    '''

    Args:
        f: the file to write on
        features_path: the path for features to be used in the prediction
        out_path: where to save the prediction
        sequence_length: the sequence length
        feat_type: which model to use
        layers: how many layers are in the given model

    Returns: a file with all the commands to be run for making predictions

    '''
    subjects = ["F", "H", "I", "J", "K", "L", "M", "N"]
    counter = 0
    for subject in subjects:
        counter = 0
        print("Running predictions for subject " + subject)
        while counter < layers:
            command = "python "+args.home_path+"/brain_language_nlp/predict_brain_from_nlp.py " \
                      " --subject "+subject+\
                      " --nlp_feat_type "+feat_type+ \
                      " --nlp_feat_dir "+features_path+ \
                      " --layer "+str(counter)+ \
                      " --sequence_length "+str(sequence_length)+ \
                      " --method " + method + \
                      " --output_dir "+out_path
            f.write(command)
            f.write("\n")
            counter += 1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_path",
                        default="/media/wrb15144/drives/i/Science/CIS-YASHMOSH/zenonlamprou/neurolinguistics-project/code/fMRI-AI-KB")
    parser.add_argument("--method", default="plain")
    parser.add_argument("--sequence_lengths", default="4")
    parser.add_argument("--models", default="bert")
    parser.add_argument("--feature_strategy", default="normal")
    args = parser.parse_args()
    print(args)

    models = args.models.split(",")
    sequence_lengths = args.sequence_lengths.split(",")
    for model in models:
        save_path = args.home_path + "/scripts/" + model + "/" + args.feature_strategy + "/" + args.method
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path += "/prediction_commands.sh"
        f = open(save_path, "w")
        for sequence_length in sequence_lengths:
            print(sequence_length)
            features_path = args.home_path + "/data/models_output/{}/features/{}/{}/".format(model,
                                                                                             args.feature_strategy,
                                                                                             sequence_length)
            out_path = args.home_path + "/data/models_output/{}/predictions/{}/{}/".format(model, args.feature_strategy,
                                                                                           sequence_length)
            layers = 13
            if model == "distilibert":
                layers = 7
            construct_commands(f, features_path, out_path, int(sequence_length), model, layers, args.method)
        f.close()