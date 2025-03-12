# SEM_grain_segmentation

I. Training the pixel classifier
Most important thing is to creat label files for training.

Option 1: Using ImageJ/Fiji
draw the grain boundaries with white color (value 255), then you can change contrast and tune the threshold to turn the gray background (value 0) completely to black. Then we get a binary train label file.
The training script will turn it into (0,1) binary label automatically.



TODO: fine-tunning the model using cross-validation on the train_image

II. Predict using the pre-trained model
in predict notebook, change the test image file path to the image you want to test on
then run all codes, it will generate a file "predict_GBs", this is a binary image (0 - black background grain area, 1 - white grain boundaries ) 
so it may seems total black, but it's fine

III. Post processing and analysis
in the postprocess_analysis notebook, change the test image and prediction image(generated in last step) file paths.
then run all codes, it will save a segmentation_result image, and grain diameter list csv file,
the grain size histogram will also be presented, but you may need to save it manually.
