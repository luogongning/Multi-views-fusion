# Multi-views fusion
LV Volumes Prediction based on Multi-views Fusion CNN
Left ventricular (LV) volumes estimation is a critical procedure for cardiac disease diagnosis. The traditional estimation methods are mainly based on image segmentation technology. In this paper, we proposed a direct volumes prediction method based on the end-to-end deep convolutional neural networks (CNN). We study the end-to-end LV volumes prediction method in the items of the data preprocessing, networks structure, and multi-views fusion strategy. The main contributions of this paper are the following aspects. First, we proposed a new data preprocessing method on CMR. Second, we proposed a new networks structure for end-to-end LV volumes estimation. Third, we explored the representational capacity of different slices, and proposed a fusion strategy to improve the prediction accuracy. The evaluation results on the open accessible benchmark datasets prove that the proposed method has higher accuracy than the state-of-the-art prediction methods in terms of end-diastole volumes (EDV), end-systole volumes (ESV), ejection fraction (EF).


This code includes the following three parts:
1. The ROI detection and data proprocessing.

In the file of ROI detection folder, the find_ventricle_location.py can achieve the LV region detection,and the *.png files are the samples results of ROI detection. *.josn files save the center points.
dataprocessing.py	is the data preprocessing code, which includes the slices selection and training data generation function.
train.py is the training code for the model training including the function of training parameters (batch size, interation number, and model saving strategy)
utils.py is the tool code including data agumentation function to achieve the data augmentation through rotation and shift operation.

2. CNN model and training code based on keras.

The Model folder save the CNN model files

3. submission and statistic

submission.py	is the testing results generation code.



If you have any questions, please do not hesitate to contact with Gongning Luo, email:luogongning@hit.edu.cn
