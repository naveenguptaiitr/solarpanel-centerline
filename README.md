This is the repository for SolarPanel Centerline detection. 

This repository explores following model for centerline detection
1. U-Net with pre-trained ResNet34 encoder
2. U-Net with pre-trained ResNet50 encoder
3. DeepLabV3 with pre-trained ResNet50 encoder
4. ConvNeXt model

Steps to install dependencies:
1. Create a pip or conda environment and activate it.
2. python3 -m pip install -r requirements.txt

To run the inference for centerline detection, please follow the steps outlined below. Of note, all trained checkpoints are stored under checkpoints directory.
1. Go to scripts and open the inference.ipynb.
2. The inference notebook has following parameters to set to allow running different models: MODEL_ARCHITECTURE, MODEL_NAME, MODEL_ENCODER, CHECKPOINT_PATH, TRAIN_FRAC, VAL_FRAC, SPLIT_STRATEGTY, START_IMAGE_IDX AND NUM_IMAGES. 
3. The notebook would also give the visualization of the model performance.


To train these model, please follow the steps outlined below.
1. Go to scripts and open the train.ipynb.
2. The train notebook has following parameters to set to allow training of different models: MODEL_ARCHITECTURE, MODEL_NAME, MODEL_ENCODER, TRAIN_FRAC, VAL_FRAC, SPLIT_STRATEGTY, SPLIT_STRATEGTY, DATA_AUGMENTATION, EPOCHS, CHECKPOINT_PATH