# Brain-Transformer-Mapping-On-Punctuation

Role of Punctuation in Semantic Mapping between Brain and Transformer Models (Published in ACAIN 2022)

## Description

Provide a short description explaining the what, why, and how of your project. Use the following questions as a guide:

- What was your motivation?
- Why did you build this project? (Note: the answer is not "Because it was a homework assignment.")
- What problem does it solve?
- What did you learn?

## Table of Contents

If your README is long, add a table of contents to make it easy for users to find what they need.

- [Installation](#installation)
- [Helper Scripts](#helper-scripts)
- [Experiment functions](#experiment-functions)
- [Credits](#credits)
- [License](#license)

## Installation
Use the ```pip install -r requirements.txt``` command to install the necessary packages on your virtual environment.
Use the ```pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113``` to install the necessary torch library.
Additionally, if you want to install a different distribution of torch visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and get the instruction for the desired torch distribution.
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



## Credits

List your collaborators, if any, with links to their GitHub profiles.

If you used any third-party assets that require attribution, list the creators with links to their primary web presence in this section.

If you followed tutorials, include links to those here as well.

## License
