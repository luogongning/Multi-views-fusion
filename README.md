# Multi-views fusion
LV Volumes Prediction based on Multi-views Fusion CNN
Left ventricular (LV) volumes estimation is a critical procedure for cardiac disease diagnosis. The traditional estimation methods are mainly based on image segmentation technology. In this paper, we proposed a direct volumes prediction method based on the end-to-end deep convolutional neural networks (CNN). We study the end-to-end LV volumes prediction method in the items of the data preprocessing, networks structure, and multi-views fusion strategy. The main contributions of this paper are the following aspects. First, we proposed a new data preprocessing method on CMR. Second, we proposed a new networks structure for end-to-end LV volumes estimation. Third, we explored the representational capacity of different slices, and proposed a fusion strategy to improve the prediction accuracy. The evaluation results on the open accessible benchmark datasets prove that the proposed method has higher accuracy than the state-of-the-art prediction methods in terms of end-diastole volumes (EDV), end-systole volumes (ESV), ejection fraction (EF).


This code includes the following parts:
1. The ROI detection and data proprocessing:

In the file of ROI detection folder, the find_ventricle_location.py can achieve the LV region detection,and the *.png files are the samples results of ROI detection. *.josn files save the center points.

2. dataprocessing.py is the data preprocessing code, which includes the slices selection and training data generation function.

3. train.py is the training code for the model training including the control of training parameters (batch size, interation number, and model saving strategy)

4. utils.py is the tool code including data agumentation function to achieve the data augmentation through rotation and shift operation.

5. CNN model and training code based on keras.

The Model folder save the common CNN model files, Model.py is the model designed freely. 

6. submission and statistic

submission.py is the testing results generation code.

7. systole_loss_processing.txt saves the validation loss after every epoch during model training, val_loss_ES.txt records the lowest validation loss value on validation set.

8. weights_systole_best.hdf5 saves the weights of model with the lowest validation loss.

# Requirements

Python3.5, Keras 

# Training

python train.py

# Testing

python submission.py

# Citation
If you use this code/model for your research, please consider citing the following paper:

```
@article{luo2018multi,
  title={Multi-views fusion CNN for left ventricular volumes estimation on cardiac MR images},
  author={Luo, Gongning and Dong, Suyu and Wang, Kuanquan and Zuo, Wangmeng and Cao, Shaodong and Zhang, Henggui},
  journal={IEEE Transactions on Biomedical Engineering},
  volume={65},
  number={9},
  pages={1924--1934},
  year={2018},
  publisher={IEEE}
}
```

```
@inproceedings{luo2016novel,
  title={A novel left ventricular volumes prediction method based on deep learning network in cardiac MRI},
  author={Luo, Gongning and Sun, Guanxiong and Wang, Kuanquan and Dong, Suyu and Zhang, Henggui},
  booktitle={2016 Computing in Cardiology Conference (CinC)},
  pages={89--92},
  year={2016},
  organization={IEEE}
}
```

If you have any questions, please do not hesitate to contact with Gongning Luo, email:luogongning@163.com
