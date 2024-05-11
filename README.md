This file comes with a Jupyter Notebook that includes the code of Training/Validation/Testing of the Neural Networks described in the Report. It may take some time to run.

In order to use the code, be sure to use the environment with the dependencies in the TorchEnv.yaml file by entering in the console (with anaconda for example):

```
conda env create -f TorchEnv.yaml
```

In case, there are any dependency errors, the file "TorchEnvSafe.yaml" contains the environment I used during my experiments.

In the directory of the Jupyter notebook, there should be a preprocessed directory with empty folders fold1, fold2, fold3,... where the preprocessed data will be stored. In case you wish to relocate the notebook, be sure to take this into account. 

In the second cell, you must include the parent directory to all the raw fold1,fold2,... folders. 
Example: you have the directory "/home/user/audio/fold1", "/home/user/audio/fold2", etc..., so you should write "/home/user/audio/" as your directory variable (take care to write the last "/").

Also, be sure to include the dataset csv file in the same directory as the notebook (this file already has one).

The deepfool.py file is a version of the code found in https://github.com/LTS4/DeepFool/blob/master/Python/deepfool.py slightly altered by me in order to use GPU capabilities and must be included in the same directory as the notebook.

# ATENTION

In the ./runs file you can find all the runs that I use to write the report included the training runs, which I don't include there. Be aware that running this code will overwrite those.

