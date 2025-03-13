# SEM_grain_segmentation

# Image Processing and Grain Boundary Detection

This project contains tools for processing microscopy images and detecting grain boundaries using machine learning.

## Setup Instructions

### Option 1: Using Conda (Recommended)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 

2. Create the environment from the environment.yml file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate mini-proj
   ```

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

### Option 2: Using pip and venv

1. Create a virtual environment:
   ```bash
   python -m venv sem_seg
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     sem_seg\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source sem_seg/bin/activate
     ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```


## Running the Notebook

1. After setup, open the `preprocess_train.ipynb` file in Jupyter
2. Make sure your image files (`train_image.tif` and `train_labels.tif`) are in the same directory
3. Run all cells to process images and train the model

## Files Description

- `preprocess_train.ipynb` - Jupyter notebook containing image preprocessing and model training code
- `best_pixel_classifier.joblib` - Trained Random Forest classifier model
- `environment.yml` - Conda environment specification
- `requirements.txt` - Pip requirements file

I. Training the pixel classifier
Most important thing is to creat label files for training.

Option 1: Using ImageJ/Fiji
draw the grain boundaries with white color (value 255), then you can change contrast and tune the threshold to turn the gray background (value 0) completely to black. Then we get a binary train label file.
The training script will turn it into (0,1) binary label automatically.

II. Predict using the pre-trained model
in predict notebook, change the test image file path to the image you want to test on
then run all codes, it will generate a file "predict_GBs", this is a binary image (0 - black background grain area, 1 - white grain boundaries ) 
so it may seems total black, but it's fine

III. Post processing and analysis
in the postprocess_analysis notebook, change the test image and prediction image(generated in last step) file paths.
then run all codes, it will save a segmentation_result image, and grain diameter list csv file,
the grain size histogram will also be presented, but you may need to save it manually.
