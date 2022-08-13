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
- [Usage](#usage)
- [Credits](#credits)
- [License](#license)

## Installation
Use the ```pip install -r requirements.txt``` command to install the necessary packages on your virtual environment.
Use the ```pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113``` to install the necessary torch library.
Additionally, if you want to install a different distribution of torch visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and get the instruction for the desired torch distribution.
## Usage
### Creating the save space
Using the script create_saving_space.py under helper_utils can help you create the save space for all your experiments.

It creates all the folders for all the available models mentioned in the paper. The only parameter needs to specified is the --sequence_length parameter.

Define all the sequence lengths like so : --sequence_lengths 4,5,10,15,20,25,30,25,40. 

IMPORTANT TO USE THE "," BETWEEN THE LENGTHS SO THE PROGRAM CAN IDENTIFY ALL THE LENGTHS!!!!

Example:
```python .\helper_utils\create_saving_space.py --sequence_lengths 4,5,10,15,20,25,30,35,40```
### Creating the extraction features script
### Creating the prediction scripts
### Creating the evaluation scripts
### Extracting the features
### Making predictions
### Evaluating predictions
Provide instructions and examples for use. Include screenshots as needed.

To add a screenshot, create an `assets/images` folder in your repository and upload your screenshot to it. Then, using the relative filepath, add it to your README using the following syntax:

    ```md
    ![alt text](assets/images/screenshot.png)
    ```

## Credits

List your collaborators, if any, with links to their GitHub profiles.

If you used any third-party assets that require attribution, list the creators with links to their primary web presence in this section.

If you followed tutorials, include links to those here as well.

## License
