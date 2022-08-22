import argparse
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_lengths", default=40)
    parser.add_argument("--home_path", default=os.getcwd())
    args = parser.parse_args()
    sequence_lengths = args.sequence_lengths.split(",")
    models = ["bert", "distilbert", "roberta", "albert", "electra"]
    feature_strategies = ["normal", "padding_all", "padding_everything", "padding_fixations", "removing_fixations"]
    home_path = args.home_path.replace("\\", "/")
    for sequence_length in sequence_lengths:
        for model in models:
            script_directory = home_path+"/data/models_output/{}/features/scripts/".format(model)
            if not os.path.exists(script_directory):
                os.mkdir(script_directory)
            for feature_strategy in feature_strategies:
                feature_strategy_script = home_path + "/data/models_output/{}/features/scripts/{}/".format(model,feature_strategy)
                if not os.path.exists(feature_strategy_script):
                    os.mkdir(feature_strategy_script)
                save_directory =  home_path + "/data/models_output/{}/features/scripts/{}/extract_features.sh".format(model,feature_strategy)
                output_dir = home_path+"/data/models_output/{}/features/{}/{}/".format(model,feature_strategy,sequence_length)
                command = "python {}/brain_language_nlp/extract_nlp_features.py --nlp_model {} --sequence_length {} --output_dir {} --home_path {} --feature_strategy {}"\
                    .format(home_path,model,sequence_length,output_dir,home_path,feature_strategy)
                with open(save_directory, "a") as f:
                    f.write(command+"\n")

