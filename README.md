# AUDIO CLASSIFICAION WITH CUSTOM DATASET
    - (BASED ON MIAI AUDIO CLASSIFICATION PROJECT)

## Dataset
Add data folder to this project, containing one train csv file and one test csv file with two columns: slice_file_name and class. One wav folder contains train and test wav audio raw data
Run preprocessing.py if you don't have spectrogram folder
    * 1 is speaker verification
    * 2 is fake voice recognition
    * 3 is command detection

## Train
    * run train_model.py for speaker verificaion task
    * run train_model_fake_task.py for fake voice recognitinon task
    * run train_model_command_detection.py for command detection task

## Testing
run test_model.py and change some lines (noted)
