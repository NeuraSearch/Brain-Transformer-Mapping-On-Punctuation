# Brain-Transformer-Mapping-On-Punctuation

Role of Punctuation in Semantic Mapping between Brain and Transformer Models (Published in ACAIN 2022)

## Description
This project has 2 goals. The first goal was to extend the work done by [Toneva and Wehbe](https://dl.acm.org/doi/abs/10.5555/3454287.3455626) on identifying brain
align models in the domain of Natural Language Processing. The second goal was to determine the role of punctuation and how is processed
semantically from the human brain.

This README file provides instructions on how to run our code and reproduce our results.
Provide a short description explaining the what, why, and how of your project. Use the following questions as a guide:


## Table of Contents

- [Installation](#installation)
- [Helper Scripts](#helper-scripts)
- [Experiment functions](#experiment-functions)

## Installation
Use the ```pip install -r requirements.txt``` command to install the necessary packages on your virtual environment.
Use the ```pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113``` to install the necessary torch library.
Additionally, if you want to install a different distribution of torch visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and get the instruction for the desired torch distribution.

You can download the already [preprocessed fMRI data here](https://drive.google.com/drive/folders/1Q6zVCAJtKuLOh-zWpkS3lH8LBvHcEOE8?usp=sharing), which are made publicly available from [Toneva and Wehbe](https://dl.acm.org/doi/abs/10.5555/3454287.3455626). 
This data contains fMRI recordings for 8 subjects reading one chapter of Harry Potter as mentioned in the original README file which can be found under brain_language_nlp directory. 
The data must be placed on the data directory in order for the code to run. 
## Helper Scripts

You can use the 4 helper scripts to construct the saving space for your data and create bash scripts to run the commands for all 3 phases of the experiment.
### Creating the save space
Using the script create_saving_space.py under helper_utils can help you create the save space for all your experiments.

It creates all the folders for all the available models mentioned in the paper. The only parameter needs to specified is the --sequence_length parameter.

#### Parameters
* **sequence_lengths** :Define all the sequence lengths like so : --sequence_lengths 4,5,10,15,20,25,30,25,40. IMPORTANT TO USE THE "," BETWEEN THE LENGTHS SO THE PROGRAM CAN IDENTIFY ALL THE LENGTHS!!!!
* **home_path** : Defines the base directory of the saving space. By default is set to be the start of the directory that the scripts are saved on. This is the same in all helper scripts.

Example:
```python .\helper_utils\create_saving_space.py --sequence_lengths 4,5,10,15,20,25,30,35,40```
### Creating the extraction features script
### Creating the prediction scripts
You can use the script construct_prediction_commands.py under helper_utils to create the scripts that can be used to make predictions using the extracted features.
#### Parameters:
* **sequence_lengths** : Define all the sequence lengths like so : --sequence_lengths 4,5,10,15,20,25,30,25,40. IMPORTANT TO USE THE "," BETWEEN THE LENGTHS SO THE PROGRAM CAN IDENTIFY ALL THE LENGTHS!!!!
* **method** :  The method to be used for the prediction processed. Can be one of these 5 options : "plain","kernel_ridge","kernel_ridge_svd","svd","ridge_sk".
* **feature_strategy** : The feature strategy used previously on extracting the features. This parameter is essential to be able to navigate to the correct directory to get the correct extraction files.
* **models**: Define for which model you wish to create the bash scripts that contain the prediction commands.

Example:
```python .\helper_utils\construct_prediction_commands.py --method kernel_ridge --feature_strategy padding_all --models bert,albert,roberta,distilibert,electra --sequnce_lengths 4,5,10,15,20,25,30,35,40```
### Creating the evaluation scripts
You can use the script construct_evaluation_commands.py under helper_utils to create the scripts that can be used to evaluate the performance of the models.
#### Parameters:
* **sequence_lengths** : Define all the sequence lengths like so : --sequence_lengths 4,5,10,15,20,25,30,25,40. IMPORTANT TO USE THE "," BETWEEN THE LENGTHS SO THE PROGRAM CAN IDENTIFY ALL THE LENGTHS!!!!
* **models**: Define for which model you wish to create the bash scripts that contain the evaluation commands.
* **feature_strategy** : The feature strategy used previously on extracting the features. This parameter is essential to be able to navigate to the correct directory to get the correct prediction files.
* **method** :  The method to be used by the prediction processed. This parameter is essential to be able to navigate to the correct directory to get the correct prediction files.
### Extracting the features
### Making predictions
### Evaluating predictions




