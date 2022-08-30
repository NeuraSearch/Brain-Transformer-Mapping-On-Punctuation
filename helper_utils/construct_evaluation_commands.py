import argparse
import os
from pathlib import Path


def construct_eval_commands(model,sequence_length,layers,home_path,method,feature_strategy):
    '''

    Args:
        model: the given model
        sequence_length: the given sequence length
        layers: number of layer
        home_path: the base path depending on the operating system

    Returns: for each layer in each length for each subject returns a bash command file. Each line is a command to be run to evaluate a different predicition file
    saves time instead of running one command at the time

    '''
    subjects = ["F", "H", "I", "J", "K", "L", "M", "N"]
    counter = 0
    while counter < layers:
        print("Saving for layer "+str(counter))
        save_path_directory = home_path + "/scripts/{}/evaluations/{}/{}/{}/{}/".format(model,feature_strategy,method,sequence_length,counter)
        if not os.path.exists(save_path_directory):
            Path(save_path_directory).mkdir(parents=True, exist_ok=True)
        print(save_path_directory)
        if not os.path.exists(save_path_directory):
            os.mkdir(save_path_directory)
        save_path = save_path_directory+"evaluation_script.sh"
        print(save_path)
        file = open(save_path,"w")
        for subject in subjects:
            output_path = home_path+f"/data/models_output/{model}/evaluations/{feature_strategy}/{method}/{sequence_length}/{counter}/"
            input_path = home_path + "/data/models_output/{}/predictions/{}/{}/{}/predict_{}_with_{}_layer_{}_len_{}.npy".format(model,feature_strategy, sequence_length,method,subject,model,counter,sequence_length)
            command = "python "+home_path+"/brain_language_nlp/evaluate_brain_predictions.py " \
                      " --subject " + subject + \
                      " --method " + method + \
                      " --input_path " + input_path + \
                      " --output_path " +output_path\

            file.write(command)
            file.write("\n")
        counter += 1
        file.close()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default='bert',choices=["bert", "distilbert", "roberta", "albert", "electra"])
    parser.add_argument("--sequence_lengths",default="4",help='length of context to provide to NLP model (default: 1)')
    parser.add_argument("--home_path", default="")
    parser.add_argument("--feature_strategy", default="normal",choices=["normal","padding_all","padding_everything","padding_fixations","removing_fixations"])
    parser.add_argument("--method", default="plain",choices=["plain","kernel_ridge","kernel_ridge_svd","svd","ridge_sk"])
    args = parser.parse_args()
    print(args)
    home_path = args.home_path
    if args.home_path == "":
        home_path = os.getcwd().replace("\\", "/")
    models = args.models.split(",")
    sequence_lengths = args.sequence_lengths.split(",")
    for model in models:
        layers = 13
        if model == "distilibert":
            layers = 7
        for sequence_length in sequence_lengths:
            construct_eval_commands(model, sequence_length, layers, home_path, args.method, args.feature_strategy)