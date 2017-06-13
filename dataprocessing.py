from __future__ import print_function

import csv
import os
import numpy as np
from numpy import mat, zeros
import pydicom as dicom
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure
from skimage.filter import canny
from skimage.filter import threshold_otsu
import json
import cv2

img_resize = True
img_shape = (224, 224)

def read_center_file(jsonname):
    json_path = os.path.join('E:', 'calc', jsonname)
    geom = dict()
    if os.path.isfile(json_path):
        f = open(json_path, 'r')
        geom = json.load(f)
        f.close()
    keys = list(geom.keys())
    for el in keys:
        geom[int(el)] = geom[el]
    for el in keys:
        geom.pop(el, None)
    return geom

def convert_to_grayscale_with_increase_brightness_fast(im, incr):
   min = np.amin(im.astype(float))
   max = np.amax(im.astype(float))
   out = incr*((im - min) * (255)) / (max - min)
   out[out > 255] = 255
   out = out.astype(np.uint8)
   return out

def crop_resize_check(dcm_path, id, slicelocalization,PixelSpacing,sax, point, imagename):
    dcm_path = dcm_path.replace('\\', '/')
    debug_folder = os.path.join('E:', 'calc', 'checkEDES')
    if not os.path.isdir(debug_folder):
        os.mkdir(debug_folder)
    ds = dicom.read_file(dcm_path)
    img = convert_to_grayscale_with_increase_brightness_fast(ds.pixel_array, 1)
    raw=int(round(point[0], 0))
    #F_col=float(col)
    col=int(round(point[1], 0))
    #F_raw=float(raw)
    row_spacing =PixelSpacing[0]
    row_spacing = float(row_spacing)
    row_spacing_size=(32.0*1.4)/row_spacing
    col_spacing = PixelSpacing[1]
    col_spacing = float(col_spacing)
    col_spacing_size=(32.0*1.4)/col_spacing
    L_raw=raw-int(row_spacing_size)
    R_raw=raw+int(row_spacing_size)
    L_col=col-int(col_spacing_size)
    R_col=col+int(col_spacing_size)
    crop_img = img[L_raw:R_raw, L_col:R_col]
    #img_shape = (64, 64)
    crop_img = imresize(crop_img, img_shape)
    crop_img=exposure.equalize_hist(crop_img)
    crop_img=crop_img*255
    crop_img=crop_img.astype(np.uint8)
    #cv2.circle(img, (int(round(point[1], 0)), int(round(point[0], 0))), 5, 255, 3)
    #cv2.circle(img, 15, 25, 5, 255, 3)
    #img = cv2.line(img, (points[1], points[0]), (points[3], points[2]), 127, thickness=2)
    #img = cv2.line(img, (points[5], points[4]), (points[7], points[6]), 127, thickness=2)
    #img = cv2.line(img, (12, 13), (112, 223), 127, thickness=2)
    #img = cv2.line(img, (12,115), (112,256), 127, thickness=2)
    #show_image(img)
    cv2.imwrite(os.path.join(debug_folder, str(id) + '_' + sax  + '_'+ imagename+'_'+str(slicelocalization)+'.jpg'), crop_img)
    return crop_img

def crop_resize(img,space):
    """
    Crop center and resize.

    :param img: image to be cropped and resized.
    """
    ws=float(space[0])
    hs=float(space[1])
    wsextend= ws/1.4
    hsextend=hs/1.4
    wsint= int(img.shape[0]*wsextend)
    if wsint%2!=0:
        wsint+=1
    hsint=int(img.shape[1]*hsextend)
    if hsint%2!=0:
        hsint+=1
    #if img.shape[0] < img.shape[1]:
        #img = img.T
    # we crop image from center
    image_shape=(wsint,hsint)
    img = imresize(img, image_shape)
    #plt.imshow(img,cmap=cm.gray)
    #short_edge = min(img.shape[:2])
    yyy=img.shape[0]
    xxx=img.shape[1]
    yy = int((img.shape[0] - cropsize) / 2)
    xx = int((img.shape[1] - cropsize) / 2)
    if (yy<=0) and (xx<=0) :
        crop_img=np.zeros((cropsize, cropsize), dtype=np.float32)
        for i in range(-yy,-yy+yyy):
            for j in range(-xx,-xx+xxx):
                crop_img[i][j]=img[i+yy][j+xx]
        img=crop_img
    elif (yy<=0) and (xx>=0) :
        crop_img=np.zeros((cropsize, cropsize), dtype=np.float32)
        for i in range(-yy,-yy+yyy):
            for j in range(0,cropsize):
                crop_img[i][j]=img[i+yy][j+xx]
        img=crop_img
    elif (yy>=0) and (xx<=0) :
        crop_img=np.zeros((cropsize, cropsize), dtype=np.float32)
        for i in range(0,cropsize):
            for j in range(-xx,-xx+xxx):
                crop_img[i][j]=img[i+yy][j+xx]
        img=crop_img
    else:
        crop_img = img[yy: yy + cropsize, xx: xx + cropsize]
        img = crop_img
    img = imresize(img, img_shape)
    return img 
  

def load_images_CH(from_dir, verbose=True):
    """
    Load images in the form study * slices * width * height.
    Each image contains 30 time series frames so that it is ready for the convolutional network.
    :param from_dir: directory with images (train or test)
    :param verbose: if true then print data
    """
    print('-'*50)
    print('Loading all DICOM images from {0}...'.format(from_dir))
    print('-'*50)
    
    current_study_sub = ''  # saves the current study sub_folder
    current_study = ''  # saves the current study folder
    current_study_images = []  # holds current study images
    ids = []  # keeps the ids of the studies
    study_to_images = dict()  # dictionary for studies to images
    study_to_age = dict()
    total = 0
    age=-1
    images = []  # saves 30-frame-images
    from_dir = from_dir if from_dir.endswith('/') else from_dir + '/'
    for subdir, _, files in os.walk(from_dir):
        subdir = subdir.replace('\\', '/')  # windows path fix
        subdir_split = subdir.split('/')
        study_id = subdir_split[-3]
        if "ch_" in subdir:
            for f in files:
                if len(files)!=30:
                    print('error {0}'.format(subdir))
                image_path = os.path.join(subdir, f)
                if not image_path.endswith('0001.dcm'):
                    if not image_path.endswith('0002.dcm'):
                        if not image_path.endswith('0003.dcm'):
                            if not image_path.endswith('0004.dcm'):
                                if not image_path.endswith('0005.dcm'):
                                    if not image_path.endswith('0006.dcm'):
                                        if not image_path.endswith('0007.dcm'):
                                            if not image_path.endswith('0008.dcm'):
                                                if not image_path.endswith('0009.dcm'):
                                                    if not image_path.endswith('0010.dcm'):
                                                        continue
                        #if not image_path.endswith('0011.dcm'):
                image = dicom.read_file(image_path)
                age_one=image.PatientAge
                space=image[0x28,0x30].value
                image = image.pixel_array.astype(float)
                #with open('shape.txt', mode='a') as f:
                    #f.write(str(image.shape[0]))
                    #f.write('\n')
                    #f.write(str(image.shape[1]))
                    #f.write('\n')
                    #f.close()
                
                image /= np.max(image)  # scale to [0,1]
                if img_resize:
                    image = crop_resize(image,space)

                if current_study_sub != subdir:
                    #x = 0
                    try:
                        if len(images) < 10:
                            print('(images) < 2 in {0} images loaded.'.format(image_path))
                            with open('eroor.txt', mode='a') as f:
                                f.write(str(image_path))
                                f.write('\n')
                            #images.append(images[x])
                            #x += 1
                        if len(images) > 10:
                            print('(images) > 2 in {0} images loaded.'.format(image_path))
                            with open('eroor.txt', mode='a') as f:
                                f.write(str(image_path))
                                f.write('\n')
                            #images = images[0:2]

                    except IndexError:
                        pass
                    current_study_sub = subdir
                    if current_study_sub!="":
                        current_study_images.append(images)  
                    images = []
                if current_study != study_id:
                    if current_study != "":
                        study_to_images[current_study] = np.array(current_study_images)
                        study_to_age[current_study] = age
                        ids.append(current_study)#avoid that the first element is '',control it through ids
                    current_study = study_id
                    current_study_images = []
                images.append(image)
                age=age_one
                if verbose:
                    if total % 1000 == 0:
                        print('Images processed {0}'.format(total))
                total += 1
    #x = 0
    try:
        if len(images) < 10:
            print('(images) < 2 in {0} images loaded.'.format(image_path))
            with open('eroor.txt', mode='a') as f:
                f.write(str(image_path))
                f.write('\n')

            #images.append(images[x])
            #x += 1
        if len(images) > 10:
            print('(images) > 2 in {0} images loaded.'.format(image_path))
            with open('eroor.txt', mode='a') as f:
                f.write(str(image_path))
                f.write('\n')            #images = images[0:3]
    except IndexError:
        pass

    print('-'*50)
    print('All DICOM in {0} images loaded.'.format(from_dir))
    print('-'*50)

    current_study_images.append(images)
    study_to_images[current_study] = np.array(current_study_images)
    if current_study != "":
        ids.append(current_study)

    return ids, study_to_images, study_to_age


def load_images_SAX_LocalizationEDES(from_dir, verbose=True):
    """
	we calculated the pixels’ intensity sum in
	ROI for every frame, and selected frames corresponding to the
	maximum and minimum intensity sums as the ED and ES frames respectively.
    """
    print('-'*50)
    print('Loading all DICOM images from {0}...'.format(from_dir))
    print('-'*50)
    
    center_train=read_center_file('center_points_train.json')
    center_test=read_center_file('center_points_test.json')

    current_study_sub = ''  # saves the current study sub_folder
    current_study = ''  # saves the current study folder
    current_study_images = []  # holds current study images
    current_study_imagestwo = []
    ids = []  # keeps the ids of the studies
    study_to_images = dict()  # dictionary for studies to images
    study_to_age = dict()
    total = 0
    saxnumber_number=-1
    images = []  # saves 30-frame-images
    saxnumber = []
    saxnumbertwo = []
    sum_on_each_frame=[]
    age=-1
    sum_index = []
    #imgbefore=mat(zeros((64,64)))

    from_dir = from_dir if from_dir.endswith('/') else from_dir + '/'
    for subdir, _, files in os.walk(from_dir):
        subdir = subdir.replace('\\', '/')  # windows path fix
        subdir_split = subdir.split('/')
        study_id = subdir_split[-3]
        study_sax = subdir_split[-1]
        saxstring = study_sax.split('_')
        with open('ESED.txt', mode='a') as f1:
            f1.write(str(study_id))
            f1.write('  ')
            f1.write(str(study_sax))
            f1.write('\n')
        f1.close
        if "sax" in subdir:            
            for f in files:
                image_path = os.path.join(subdir, f)
                image = dicom.read_file(image_path)
                idone=int(study_id)
                print(idone)
                point=[]
                if idone<=700:
                    point=center_train[idone][study_sax]
                #point=center_train[idone][study_sax]
                print(image_path)
                if idone>700:
                    point=center_test[idone][study_sax]
                    print(point)
                image = image.pixel_array.astype(float)
                imageN=np.array(image)
                sum_image = imageN.sum()
                if current_study_sub != subdir:
                    
                    if current_study_sub!='':
                        a=sum_index
                        sum_on_each_frame.append(a)
                        sum_index=[]
                    current_study_sub=subdir

                if current_study != study_id :
                    
                    if current_study != '':
                        imagearray=np.array(sum_on_each_frame)
                        imagearray_axis=imagearray.sum(axis=0)
                        imagearray_axis=imagearray_axis.tolist()
                        print(imagearray_axis.index(min(imagearray_axis)))
                        print(min(imagearray_axis))
                        print(imagearray_axis.index(max(imagearray_axis)))
                        print(max(imagearray_axis))
                        with open('ESED.txt', mode='a') as f1:
                            f1.write(str(imagearray_axis.index(min(imagearray_axis))))
                            f1.write(': ')
                            f1.write(str(min(imagearray_axis)))
                            f1.write(' ')
                            f1.write(str(imagearray_axis.index(max(imagearray_axis))))
                            f1.write(': ')
                            f1.write(str(max(imagearray_axis)))
                            f1.write('\n')
                        f1.close
                        sum_on_each_frame=[]
                    current_study=study_id

                sum_index.append(sum_image)
                
    #x = 0
    return ids, study_to_images, study_to_age

def load_images_SAX(from_dir, verbose=True):
    """
    Load images in the form study * slices * width * height.
    Each image contains 30 time series frames so that it is ready for the convolutional network.
    :param from_dir: directory with images (train or test)
    :param verbose: if true then print data
    """
    print('-'*50)
    print('Loading all DICOM images from {0}...'.format(from_dir))
    print('-'*50)
    
    center_train=read_center_file('center_points_train.json')
    center_test=read_center_file('center_points_test.json')

    current_study_sub = ''  # saves the current study sub_folder
    current_study = ''  # saves the current study folder
    current_study_images = []  # holds current study images
    current_study_imagestwo = []
    ids = []  # keeps the ids of the studies
    study_to_images = dict()  # dictionary for studies to images
    study_to_age = dict()
    total = 0
    saxnumber_number=-1
    images = []  # saves 30-frame-images
    saxnumber = []
    saxnumbertwo = []
    age=-1
    from_dir = from_dir if from_dir.endswith('/') else from_dir + '/'
    for subdir, _, files in os.walk(from_dir):
        subdir = subdir.replace('\\', '/')  # windows path fix
        subdir_split = subdir.split('/')
        study_id = subdir_split[-3]
        study_sax = subdir_split[-1]
        saxstring = study_sax.split('_')        
        if "sax" in subdir:
            if len(files)!=30:
                print('error{0}'.format(subdir))

            for f in files:
                image_path = os.path.join(subdir, f)
                if not image_path.endswith('0001.dcm'):
                    if not image_path.endswith('0002.dcm'):
                        if not image_path.endswith('0003.dcm'):
                            if not image_path.endswith('0004.dcm'):
                                if not image_path.endswith('0005.dcm'):
                                    if not image_path.endswith('0006.dcm'):
                                        if not image_path.endswith('0007.dcm'):
                                            if not image_path.endswith('0008.dcm'):
                                                if not image_path.endswith('0009.dcm'):
                                                    if not image_path.endswith('0010.dcm'):
                                                        continue
                        #continue#if not image_path.endswith('0011.dcm'):
                image = dicom.read_file(image_path)
                #image = dicom.read_file('IM-12041-0001.dcm')
                age_one=image.PatientAge
                slicelocation=image.SliceLocation
                idone=int(study_id)
                #print(idone)
                point=[]
                if idone<=700:
                    point=center_train[idone][study_sax]
                #point=center_train[idone][study_sax]
                #print(image_path)
                if idone>700:
                    point=center_test[idone][study_sax]
                    #print(point)
                space=image[0x28,0x30].value
                #image = image.pixel_array.astype(float)
                
                #image /= np.max(image)  # scale to [0,1]
                
                if img_resize:
                    image = crop_resize_check(image_path,study_id,slicelocation,space,study_sax,point,f)
                    #plt.imshow(image,cmap=cm.gray)
                
                if current_study_sub != subdir:
                    try:
                        if len(images) < 10:
                            print('(images) < 2 in {0} images loaded.'.format(image_path))
                            with open('eroor.txt', mode='a') as f:
                                f.write(str(image_path))
                                f.write('\n')
                        if len(images) > 10:
                            print('(images) > 2 in {0} images loaded.'.format(image_path))
                            with open('eroor.txt', mode='a') as f:
                                f.write(str(image_path))
                                f.write('\n')
                    except IndexError:
                        pass
                    if current_study_sub!="" and saxnumber_number!=-1:
                        saxnumber.append(saxnumber_number) #first time number added to list,we can sort the slices according to the file numbers
                        saxnumber.sort()
                        current_study_imagestwo.append(images)
                        saxnumbertwo.append(saxnumber_number)
                    current_study_sub = subdir
                    images = []

                if current_study != study_id:
                    #aaaa=[]
                    if current_study != "":
                        count=1
                        length_sax=len(saxnumber)-2
                        interval_sample=int(length_sax/5)
                        if interval_sample<1:
                            interval_sample=1
                            print('sample error{0}'.format(current_study))
                        if len(saxnumber)<7:
                            print('error{0}'.format(current_study))
                    #for numberx in saxnumber:
                        for numberx in range(1,length_sax+5,interval_sample):
                            if count<=5:
                                current_study_images.append(current_study_imagestwo[saxnumbertwo.index(saxnumber[numberx])])
                                count+=1                                        
                        saxnumber=[]#if patent changed,
                        saxnumbertwo = []
                    if current_study != "":
                        study_to_images[current_study] = np.array(current_study_images)
                        study_to_age[current_study] = age
                        ids.append(current_study)
                    current_study = study_id
                    current_study_images = []
                    current_study_imagestwo = []
                images.append(image)
                saxnumber_number=int(saxstring[-1])#save number
                age=age_one
                if verbose:
                    if total % 1000 == 0:
                        print('Images processed {0}'.format(total))
                total += 1
    #x = 0
    try:
        if len(images) < 10:
            print('(images) < 2 in {0} images loaded.'.format(image_path))
            with open('eroor.txt', mode='a') as f:
                f.write(str(image_path))
                f.write('\n')
            #images.append(images[x])
            #x += 1
        if len(images) > 10:
            print('(images) > 2 in {0} images loaded.'.format(image_path))
            with open('eroor.txt', mode='a') as f:
                f.write(str(image_path))
                f.write('\n')
            #images = images[0:3]
    except IndexError:
        pass

    print('-'*50)
    print('All DICOM in {0} images loaded.'.format(from_dir))
    print('-'*50)

    #current_study_images.append(images)
    current_study_imagestwo.append(images)
    saxnumber.append(saxnumber_number)
    saxnumber.sort()
    saxnumbertwo.append(saxnumber_number)
    
    count=1
    length_sax=len(saxnumber)-2
    interval_sample=int(length_sax/5)
    if interval_sample<1:
        interval_sample=1
        print('sample error{0}'.format(current_study))
    if len(saxnumber)<7:
        print('error{0}'.format(current_study))
    for numberx in range(1,length_sax+5,interval_sample):
        if count<=5:
            current_study_images.append(current_study_imagestwo[saxnumbertwo.index(saxnumber[numberx])])
            count+=1
    
    if current_study != "":
        study_to_images[current_study] = np.array(current_study_images)
        study_to_age[current_study] = age
        ids.append(current_study)

    return ids, study_to_images, study_to_age


def map_studies_results():
    """
    Maps studies to their respective targets.
    """
    id_to_results = dict()
    train_csv = open('data/train.csv')
    lines = train_csv.readlines()
    i = 0
    for item in lines:
        if i == 0:
            i = 1
            continue
        id, systole, diastole = item.replace('\n', '').split(',')
        id_to_results[id] = [float(systole), float(diastole)]

    return id_to_results

def write_train_npy():
    """
    Loads the training data set including X and y and saves it to .npy file.
    """
    print('-'*50)
    print('Writing training data to .npy file...')
    print('-'*50)

    study_ids_sax, images_SAX, images_age_axi = load_images_SAX('data/train')  # load images and their ids
    study_ids_ch, images_CH, images_age_ch = load_images_CH('data/train')
    studies_to_results = map_studies_results()  # load the dictionary of studies to targets
    X_ED = []
    y = []

    for study_id in study_ids_sax:
        study_SAX = images_SAX[study_id]
        study_CH = images_CH[study_id]
        outputs = studies_to_results[study_id]
        if study_SAX.shape[0]!=5:
            print('SAX_length error processed {0}'.format(study_id))
        if study_CH.shape[0]!=2:
            print('CH_length error processed {0}'.format(study_id))
        #y.append(outputs)
        V_ED=[]
        #V_ES=[]
        #for i in range(study.shape[0]):
        dt=(outputs[1]-outputs[0])/9.0
        with open('volume.txt', mode='a') as f1:
            f1.write('\n')            
            f1.write(str(study_id))
            f1.write('ED: ')
            f1.write(str(outputs[1]))
            f1.write('ES: ')
            f1.write(str(outputs[0]))
            f1.write('dt: ')
            f1.write(str(dt))
            f1.write('\n')
        for i in range(10):
            V_ED.append(study_CH[0, i, :, :])
            V_ED.append(study_CH[1, i, :, :])
            V_ED.append(study_SAX[0, i, :, :])
            V_ED.append(study_SAX[1, i, :, :])
            V_ED.append(study_SAX[2, i, :, :])
            V_ED.append(study_SAX[3, i, :, :])
            V_ED.append(study_SAX[4, i, :, :])
            if len(V_ED)!=7:
                print('ED shape {0}'.format(len(V_ED)))
            X_ED.append(V_ED)
            V_ED=[]
            outputs_now=outputs[1]-i*dt
            with open('volume.txt', mode='a') as f1:
                f1.write(': ')
                f1.write(str(outputs_now))
            y.append(outputs_now)


    X_ED = np.array(X_ED, dtype=np.uint8)
    y = np.array(y)
    np.save('data/X_train_ALL.npy', X_ED)
    np.save('data/y_train.npy', y)
    print('Done.{0}'.format(len(X_ED)))
    
    print('Writing train set age to file...')
    fi = csv.reader(open('data/train_age_shape.csv'))
    f = open('submission_train_age.csv', 'w')
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(fi.next())
    for line in fi:
        idx = line[0]
        key = idx
        out = [idx]
        if key in images_age_axi:
            out.extend(images_age_axi[key])
        else:
            print('Miss {0}'.format(idx))
        fo.writerow(out)
    f.close()
    #print('Done.{0}'.format(len(images_age_axi)))


def write_train_npy_test():
    """
    only ED and ES phases are added into datasets
    Loads the training data set including X and y and saves it to .npy file.
    """
    print('-'*50)
    print('Writing training data to .npy file...')
    print('-'*50)

    study_ids_sax, images_SAX, images_age_axi = load_images_SAX('data/train')  # load images and their ids
    study_ids_ch, images_CH, images_age_ch = load_images_CH('data/train')
    studies_to_results = map_studies_results()  # load the dictionary of studies to targets
    X_ED = []
    y = []

    for study_id in study_ids_sax:
        study_SAX = images_SAX[study_id]
        study_CH = images_CH[study_id]
        outputs = studies_to_results[study_id]
        if study_SAX.shape[0]!=5:
            print('SAX_length error processed {0}'.format(study_id))
        if study_CH.shape[0]!=2:
            print('CH_length error processed {0}'.format(study_id))
        #y.append(outputs)
        V_ED=[]
        #V_ES=[]
        #for i in range(study.shape[0]):
        V_ED.append(study_CH[0, 0, :, :])
        V_ED.append(study_CH[1, 0, :, :])
        V_ED.append(study_SAX[0, 0, :, :])
        V_ED.append(study_SAX[1, 0, :, :])
        V_ED.append(study_SAX[2, 0, :, :])
        V_ED.append(study_SAX[3, 0, :, :])
        V_ED.append(study_SAX[4, 0, :, :])
        if len(V_ED)!=7:
            print('ED shape {0}'.format(len(V_ED)))    
        X_ED.append(V_ED)
        V_ED=[]
    
        V_ED.append(study_CH[0, 9, :, :])
        V_ED.append(study_CH[1, 9, :, :])
        V_ED.append(study_SAX[0, 9, :, :])
        V_ED.append(study_SAX[1, 9, :, :])
        V_ED.append(study_SAX[2, 9, :, :])
        V_ED.append(study_SAX[3, 9, :, :])
        V_ED.append(study_SAX[4, 9, :, :])
        if len(V_ED)!=7:
            print('ED shape {0}'.format(len(V_ED)))
        X_ED.append(V_ED)
        V_ED=[]
        y.append(outputs[1])
        #print('label ID {0}'.format(outputs[1]))
        #print('label ID {0}'.format(outputs[1]/5))
        #print('label ID {0}'.format(int(outputs[1]/5)))
        y.append(outputs[0])
        #print('label ID {0}'.format(outputs[0]))
        #print('label ID {0}'.format(outputs[0]/5))
        #print('label ID {0}'.format(int(outputs[0]/5)))


    X_ED = np.array(X_ED, dtype=np.uint8)
    y = np.array(y)
    np.save('data/X_train_ALL.npy', X_ED)
    np.save('data/y_train.npy', y)
    print('Done.{0}'.format(len(X_ED)))
    
    print('Writing train set age to file...')
    fi = csv.reader(open('data/train_age_shape.csv'))
    f = open('submission_train_age.csv', 'w')
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(fi.next())
    for line in fi:
        idx = line[0]
        key = idx
        out = [idx]
        if key in images_age_axi:
            out.extend(images_age_axi[key])
        else:
            print('Miss {0}'.format(idx))
        fo.writerow(out)
    f.close()
    #print('Done.{0}'.format(len(images_age_axi)))

check('data/train')
write_train_npy()
load_images_SAX_LocalizationEDES('data/train')
write_train_npy_test()