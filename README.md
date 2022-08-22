# Brain-Transformer-Mapping-On-Punctuation

Role of Punctuation in Semantic Mapping between Brain and Transformer Models (Published in ACAIN 2022)
# Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Helper Scripts](#helper-scripts)
- [Using the experiment functions](#using-the-experiment-functions)
# Description
This project has 2 goals. The first goal was to extend the work done by [Toneva and Wehbe](https://dl.acm.org/doi/abs/10.5555/3454287.3455626) on identifying brain
align models in the domain of Natural Language Processing. The second goal was to determine the role of punctuation and how is processed
semantically from the human brain.

This README file provides instructions on how to run our code and reproduce our results.
Provide a short description explaining the what, why, and how of your project. Use the following questions as a guide:
Click [this](#Installation) link to install the required packages



# Installation
Use the ```pip install -r requirements.txt``` command to install the necessary packages on your virtual environment.
Use the ```pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113``` to install the necessary torch library.
Additionally, if you want to install a different distribution of torch visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and get the instruction for the desired torch distribution.

You can download the already [preprocessed fMRI data here](https://drive.google.com/drive/folders/1Q6zVCAJtKuLOh-zWpkS3lH8LBvHcEOE8?usp=sharing), which are made publicly available from [Toneva and Wehbe](https://dl.acm.org/doi/abs/10.5555/3454287.3455626). 
This data contains fMRI recordings for 8 subjects reading one chapter of Harry Potter as mentioned in the original README file which can be found under brain_language_nlp directory. 
The data must be placed on the data directory in order for the code to run. 
# Helper Scripts

You can use the 4 helper scripts to construct the saving space for your data and create bash scripts to run the commands for all 3 phases of the experiment.
### Creating the save space
Using the script create_saving_space.py under helper_utils can help you create the save space for all your experiments.

It creates all the folders for all the available models mentioned in the paper. The only parameter needs to specified is the --sequence_length parameter.

#### Parameters
* **sequence_lengths** :Define all the sequence lengths like so : --sequence_lengths 4,5,10,15,20,25,30,25,40. IMPORTANT TO USE THE "," BETWEEN THE LENGTHS SO THE PROGRAM CAN IDENTIFY ALL THE LENGTHS!!!!
* **home_path** : Defines the base directory of the saving space. By default is set to be the start of the directory that the scripts are saved on. This is the same in all helper scripts.

Example:
```python
python .\helper_utils\create_saving_space.py 
--sequence_lengths 4,5,10,15,20,25,30,35,40
```
### Creating the extraction features script
You can use the script construct_extraction_features_script.py under helper_utils to construct the bash script that will run the extraction features script.
This file creates under the model folder in models_output a bash script file for every method that can run simultaneously all the lengths specified for the given sequence length.
#### Parameters:
* **sequence_lengths** :Define all the sequence lengths like so : --sequence_lengths 4,5,10,15,20,25,30,25,40. IMPORTANT TO USE THE "," BETWEEN THE LENGTHS SO THE PROGRAM CAN IDENTIFY ALL THE LENGTHS!!!!
* **home_path** : Defines the base directory of the saving space. By default is set to be the start of the directory that the scripts are saved on. This is the same in all helper scripts.

Example : 

```python
python .\helper_utils\construct_extraction_features_script.py
--sequence_lengths 4,5,10,15,20,25,30,35,40 
```
### Creating the prediction scripts
You can use the script construct_prediction_commands.py under helper_utils to create the scripts that can be used to make predictions using the extracted features.
#### Parameters:
* **sequence_lengths** : Define all the sequence lengths like so : --sequence_lengths 4,5,10,15,20,25,30,25,40. IMPORTANT TO USE THE "," BETWEEN THE LENGTHS SO THE PROGRAM CAN IDENTIFY ALL THE LENGTHS!!!!
* **method** :  The method to be used for the prediction processed. Can be one of these 5 options : "plain","kernel_ridge","kernel_ridge_svd","svd","ridge_sk".
* **feature_strategy** : The feature strategy used previously on extracting the features. This parameter is essential to be able to navigate to the correct directory to get the correct extraction files.
* **models**: Define for which model you wish to create the bash scripts that contain the prediction commands. --models bert,roberta,distilbert. IMPORTANT TO USE THE "," BETWEEN THE LENGTHS SO THE PROGRAM CAN IDENTIFY ALL THE LENGTHS!!!!

Example:
```python 
python .\helper_utils\construct_prediction_commands.py 
--method kernel_ridge 
--feature_strategy padding_all 
--models bert,albert,roberta,distilibert,electra 
--sequnce_lengths 4,5,10,15,20,25,30,35,40
```
### Creating the evaluation scripts
You can use the script construct_evaluation_commands.py under helper_utils to create the scripts that can be used to evaluate the performance of the models.
#### Parameters:
* **sequence_lengths** : Define all the sequence lengths like so : --sequence_lengths 4,5,10,15,20,25,30,25,40. IMPORTANT TO USE THE "," BETWEEN THE LENGTHS SO THE PROGRAM CAN IDENTIFY ALL THE LENGTHS!!!!
* **models**: Define for which model you wish to create the bash scripts that contain the evaluation commands. --models bert,roberta,distilbert. IMPORTANT TO USE THE "," BETWEEN THE LENGTHS SO THE PROGRAM CAN IDENTIFY ALL THE LENGTHS!!!!
* **feature_strategy** : The feature strategy used previously on extracting the features. This parameter is essential to be able to navigate to the correct directory to get the correct prediction files.
* **method** :  The method to be used by the prediction processed. This parameter is essential to be able to navigate to the correct directory to get the correct prediction files.

Example:
```python
python .\helper_utils\construct_evaluation_commands.py 
--models bert 
--sequence_lengths 4,5,10,15,20,25,30,35,40 
--feature_strategy normal 
--method plain
```


# Using the experiment functions
### Extracting the features
You can extract the features of the model by first running the constructed bash script for the extraction features. Navigate at scripts/{model_name}/features/{feature_strategy}/ and run the bash script.

Example:
```bash
cd scripts/bert/features/scripts/padding_all/
./extract_features.sh
```
The other way is to run directly each command one by one.

Example : 

```python
python {home_path}/brain_language_nlp/extract_nlp_features.py
--nlp_model bert 
--sequence_length 4 
--output_dir {home_path}/data/models_output/bert/features/normal/4/
--home_path {home_path}
--feature_strategy normal
```
### Making predictions
The easiest way to make predictions is by navigating at scripts/{model_name}/{feature_strategy}/{method}/ and find the prediction_commands.sh file. Run the commands the file to make the predictions. 

Example :
```bash 
./scripts/bert/padding_all/kernel_ridge/prediction_commands.sh
```
However, you can construct your own commands and run them one by one. Ensuring that you use {home_path} parameter can make you run the script from anywhere and not just the top of the project.

Example: 

```python
python {home_path}/brain_language_nlp/predict_brain_from_nlp.py  
--subject F 
--nlp_feat_type bert 
--nlp_feat_dir {home_path}/data/models_output/bert/features/normal/4/ 
--layer 0 
--sequence_length 4 
--method plain 
--output_dir {home_path}/data/models_output/bert/predictions/normal/4/
```


### Evaluating predictions
Evaluation can be run using 2 ways. Either running the created evaluation scripts directly or using the evaluation_original_code_but_faster.py script.

The difference is that with the second method and utilizing the Numba library the process is speed up significantly. It is a requirement to create the evaluation script before using both ways.

Example on using the created scripts directly: 
```bash 
./scripts/{model_name}/evaluations/{feature_strategy/{method}/{sequence_length}/{layer}/evaluation_script.sh
```
#### Instructions on using the evaluation_original_code_but_faster.py script
Using the example provided bellow you can run the evaluation using the fast script. However, the evaluation scripts needs to be constructed first.

Example:
```python
python ./evaluation_original_code_but_faster.py 
--sequence_lengths 4,5,10,15,20,25,30,25,40
--nlp_model bert 
--feature_strategy normal 
--method plain 
--output_dir ./data/models_output/bert/evaluations/normal/plain/4/0/
```
The third and final way to run the evaluations is to do it manually using the non-fast script.

Example: 

```python
python {home_path}/brain_language_nlp/evaluate_brain_predictions.py
--subject F 
--method plain 
--input_path {home_path}/data/models_output/bert/predictions/normal/4/plain/predict_F_with_bert_layer_0_len_4.npy
--output_path {home_path}/data/models_output/bert/evaluations/normal/plain/4/0/
```


