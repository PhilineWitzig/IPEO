### Structure
This repository contains the code for the project "The Spread of The Bark Beetle in European Forests" of the 2020/2021 IPEO class at EPFL.
It implements an image processing pipeline which can be used to execute various image analysis tasks related to monitoring the development and impact of the bark beetle in satellite imagery. Change to the folder [project](project) and run the code contained in [processing.py](project/processing.py) with the following command: 
```
python3 processing.py --branch "branch_to_execute"
```
where branch_to_execute should be either IDA or morphology. For an explanation of the two branches, read the report contained in this folder. Note that we didn't include the trained SVM due to its size. You can simply train it from scratch by running

```
python3 train_svm.py"
```
Don't forget to download the training dataset and store it in the data folder.
