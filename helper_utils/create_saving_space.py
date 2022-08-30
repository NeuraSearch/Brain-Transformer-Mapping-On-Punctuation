import argparse
import os

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
def create_save_spaces_for_features(base_dir):
    create_path(base_dir+"/features/")
    for method in methods:
        create_path(base_dir + "/features/{}/".format(method))
        for sequence_length in sequence_lengths:
            create_path(base_dir + "/features/{}/{}/".format(method,sequence_length))


def create_save_spaces_for_predictions(base_dir):
    create_path(base_dir + "/predictions/")
    for method in methods:
        create_path(base_dir + "/predictions/{}/".format(method))
        for sequence_length in sequence_lengths:
            create_path(base_dir + "/predictions/{}/{}/".format(method,sequence_length))
            for prediction_method in prediction_methods:
                create_path(base_dir + "/predictions/{}/{}/{}/".format(method, sequence_length,prediction_method))


def create_save_spaces_for_evaluations(base_dir,layers):
    create_path(base_dir + "/evaluations/")
    for method in methods:
        create_path(base_dir + "/evaluations/{}/".format(method))
        for prediction_method in prediction_methods:
            create_path(base_dir + "/evaluations/{}/{}/".format(method,prediction_method))
            for sequence_length in sequence_lengths:
                create_path(base_dir + "/evaluations/{}/{}/{}/".format(method,prediction_method, sequence_length))
                for layer in range(layers):
                    create_path(base_dir + "/evaluations/{}/{}/{}/{}/".format(method, prediction_method,
                                                                                       sequence_length,layer))
def create_saving_spaces(base_dir):
    if not os.path.exists(base_dir+"/data/"):
        os.mkdir(base_dir+"/data/")
    if not os.path.exists(base_dir+"/data/models_output/"):
        os.mkdir(base_dir+"/data/models_output/")
    available_models = ["bert","roberta","electra","albert","distilibert"]
    for model in available_models:
        create_path(base_dir+"/data/models_output/"+model)
        create_save_spaces_for_features(base_dir+"/data/models_output/"+model)
        create_save_spaces_for_predictions(base_dir+"/data/models_output/"+model)
        layers = 13
        if model == "distlibert":
            layers = 7
        create_save_spaces_for_evaluations(base_dir+"/data/models_output/"+model,layers)
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_lengths", default="4")
    parser.add_argument("--home_path", default="")
    args = parser.parse_args()
    home_path = args.home_path
    if args.home_path == "":
        home_path = os.getcwd().replace("\\", "/")
    sequence_lengths = args.sequence_lengths.split(",")
    prediction_methods = ["kernel_ridge", "kernel_ridge_svd", "plain", "ridge_sk", "svd"]
    methods = ["normal", "padding_all", "padding_everything", "removing_fixations", "padding_fixations"]
    subjects = ["F", "H", "I", "J", "K", "L", "M", "N"]
    create_saving_spaces(home_path)