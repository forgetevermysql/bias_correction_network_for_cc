# Project structure
The code for this project is based on the [matteo-rizzo](https://github.com/matteo-rizzo/fc4-pytorch) implementation of the Pytroch version of the FC4 implementation. The repository uploaded the core code for this article, you need to build the project according to the directory structure provided by matteo-rizzo.
```commandline
auxiliary
    settings.py
    utils.py
classes
    core
    data
    losses
    fc4             (Replace with the file bias_corr_cc in the repository)
dataset
    coordinates
    folds.mat
    img2npy.py
    metadata.txt
test
    test.py         (Replacement with a file from the repository)
train
    train.py        (Replacement with a file from the repository)
```
# Requirements
This project has been developed and tested using Python 3.8 and Torch > 1.7. Please install the required packages using ```pip install -r requirements.txt```


# Datasets

[Shi's Re-processing of Gehler's Raw Color Checker Dataset.](https://www2.cs.sfu.ca/~colour/data/shi_gehler/)

[NUS-8 camera dataset](https://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html)

### Dataset Catalog Structure
```commandline
dataset
    coordinates (MCC coordinates)
    images (Original image)
    folds.mat (Index of images included in cross validation)
    metadata.txt (Real Illumination Label)
```

### Preprocessing
To preprocess the dataset, run the following commands:
```commandline
python3 img2npy.py
```

### Pretrained models
Pre-trained models are located in ```./classes/bias_corr_cc/BCCCPretrained```
Please load the pre-trained model of the corresponding fold during training.
If you need to restart training, please run the file ```. /classes/bias_corr_cc/BCCCModel.py``` and comment out the following:
```self.load_pretrained_except_final_convs()```

Note: This loading pre-trained model can only be loaded with nobias (a backbone in the paper)
# Train
To train the model, run ```python train/train.py```. The training procedure can be configured by editing the value of the global variables at the beginning of the ```train.py``` file.
### global variables
```commandline
RANDOM_SEED = 0
EPOCHS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 0.0003

 # Fold num of Color Checker dataset, Valid only if ColorChecker is set to True
FOLD_NUM = 0 

# True: Color Checker dataset, False: NUS-8 camera dataset
ColorChecker = True

# NUS-8 camera dataset corresponding to eight cameras, , Valid only if ColorChecker is set to False
DATASET_NUM = 8

# Load model checkpoint and continue training
RELOAD_CHECKPOINT = False

# Loading double bias pre-trained models
RELOAD_PRETRAINED = False
```

# Test
To test the model, run ```python3 test/test.py```. The test procedure can be configured by editing the value of the global variables at the beginning of the ```test.py``` file.

# Reproduce the results
The trained model has been uploaded to A. Please download the model folder and put it in the root directory with the following structure:
```commandline
root
    trained_models
        baseline
            bccc
                fold_0
                fold_1
                fold_2
```
