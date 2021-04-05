# Text-Summarizer

## Requirements
* Python3
* Tensorflow >= 1.4 (tested on Tensorflow 1.4.1)
* numpy
* pandas
* keras
* seaborn
* spacy
* tqdm
* sumeval

You can use the python package manager of your choice (pip/conda) to install the dependencies.

# Instructions
* Download Dataset

    Please follow the instructions [here](https://github.com/abisee/cnn-dailymail) for downloading and preprocessing the CNN/DailyMail dataset. After that, copy cnn folder containing stories folder into the project's dataset directory,.

* Preprocessing the Dataset
	**Please download glove.42B.300d.txt from [here](https://www.kaggle.com/yutanakamura/glove42b300dtxt) before running preprocessing.py
	
	```
	python3 preprocessing.py
    ```

* Train the full model
    
    ```
   	python3 Model_training.py –NE False/True –encoder_epoch integer –decoder_epoch integer
    ```

* Evaluate
    
    ```
	python3 Model_evaluation.py –NE False/True –encoder_name name_of_encoder_h5_file  –decoder_name name_of_decoder_h5_file
    ```