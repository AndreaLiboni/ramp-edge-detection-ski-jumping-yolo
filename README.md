# ramp-edge-detection-ski-jumping-yolo
A detector of the edge line of the ramp in the Ski Jumping sport, using an implementation of YOLOv11

## Getting Started
Create the virtual enviroment and source into it:
```bash
$ python -m venv venv
$ source venv/bin/activate
```
Install the required packages:
```bash
$ pip install -r requirements.txt
```

## Generate the dataset
This step require to already have an export in the CVAT format with the data.
Use the dataset_convertion.py file to prepare the dataset, with the following options:
```bash
--root ROOT             root folder for the CVAT dataset
--output OUTPUT         output directory for the YOLO dataset (default: ./datasets)
--aug_level AUG_LEVEL   augmentation level for the train dataset (default: 1)
```
Note: The level 1 of augmentation means that the augmentation will not be applied.

## Use the model
To use the model you can use the model.py script with the follwing options:
```bash
--data DATA                 data.yaml path (can be set also in the congif.yaml file)
--output OUTPUT             output directory for the model (default: ./yolo_output)
--model MODEL               path to the model.pt file
--train                     start the training on the dataset
--test                      start the testing on the dataset
--forward FORWARD           evaluate the model on a video, or use the keywork 'train', 'val' and 'test' to evaluate on those set
--show_dataset              show the dataset
--plot_result PLOT_RESULT   plot the results from the results.csv YOLO file
```
If you want to change training parameters you direcly change the training args starting from line 80 of model.py
