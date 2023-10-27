#!/usr/bin/env python
# coding: utf-8

# In[1]:


# python resample.py $IMAGE_TO_RESAMPLE $REFERENCE_IMAGE
#!jupyter nbconvert classify-2023-05-11.ipynb --to python


# In[2]:


"""
~~~ Script Description ~~~
Copied from 5-12
- implemement cross condition testing
- modified so that algorithm is now correct 

"""


# ** IMPORTS **
# BASE
import numpy as np 
import os
import os.path
import scipy.io
import warnings
import sys  
import random
import deepdish as dd
import random
import pandas as pd
import scipy.io
from scipy import stats
from scipy.stats import norm, zscore, pearsonr
from scipy.signal import gaussian, convolve
from sklearn import decomposition
from sklearn.model_selection import LeaveOneOut, KFold
import sklearn.metrics 
from scipy.stats import stats
from scipy.spatial.distance import squareform
import statistics


# FMRI
import nibabel as nib
from nilearn.input_data import NiftiMasker , MultiNiftiMasker, NiftiLabelsMasker
import nilearn as nil
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_epi_mask
# Visualize it as an ROI
from nilearn.plotting import plot_roi
#plot_roi(x)
from nilearn.image import concat_imgs, resample_img, mean_img
from nilearn.plotting import view_img
from nilearn import datasets, plotting
from nilearn.input_data import NiftiSpheresMasker
from nilearn import input_data
from nilearn.plotting import plot_glass_brain
from nilearn.masking import apply_mask
from nilearn.image import concat_imgs, resample_img, mean_img,index_img
from nilearn import masking
from nilearn.plotting import view_img
from nilearn.image import resample_to_img
from nilearn import image

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import brainiak.eventseg.event
from brainiak import image, io




# Import machine learning libraries
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import LeavePGroupsOut
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy.stats import sem
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import LeaveOneGroupOut


random.seed(10)

# # Functions 

# In[4]:


#from utils import label_lists, find_cond_index, org_bold_data,load_epi_data, load_roi_mask, intersect_mask


# In[5]:


def load_roi_mask(rois_dir, ROI_name, space):
    if space == "MNI":
        maskdir = os.path.join(rois_dir)    
        print("expected shape: 78, 93,65")
    elif space == "T1":
        maskdir = os.path.join(rois_dir+ "/T1")
        print("expected shape: 56, 72,53")
    else:
        print("wrong mask input. check this function")
    # load the mask
    maskfile = os.path.join(maskdir, "%s.nii" % (ROI_name))
    mask = nib.load(maskfile)
    print("mask shape: ", mask.shape)
    print("Loaded %s mask" % (ROI_name))
    return mask


# In[6]:
      


# # New classifier functions

# ## GroupKFold classifier, n_splits = 5

# In[7]:

## norm is true because...? sam said so i think... normalizing two dif conditions i think
def clf_GroupKFold_roc_cross(X1, y1, X2, y2, groups_train, groups_test, n_splits, norm=True):
    """
    We must split data into two different train / test sets becaz there are diff number of button presses
    for [SM, SC] and [OM, OC]
    """
    # split into 5 groups # 
    group_kfold = GroupKFold(n_splits)
    group_kfold.get_n_splits(X1, y1, groups_train)
 
    
    ## set vars ## 
    clf_score = np.array([])
    roc_score = np.array([])
    inner_clf_score = np.array([])
    C_best = []

    ## Iterate through each fold ## 
    for cond_set1, cond_set2 in zip(group_kfold.split(X1,y1, groups_train), group_kfold.split(X2,y2, groups_test)):        
        # cond_set 1 is an array of len 2 consisting of train indices, test idices, for the first group 
        train_ind = 0
        test_ind = 1
        train_index = cond_set1[train_ind]
        test_index = cond_set2[test_ind]
        ## split data appropriately into five different train / test splits
        X_train = X1[train_index]
        y_train = y1[train_index]
        X_test = X2[test_index]
        y_test = y2[test_index]
        print(f'x train: {X_train.shape} y_train: {y_train.shape} x_test:{X_test.shape} y_test: {y_test.shape}')
        
        ## convert y's to arrays ## 
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        ## Further split into training groups for grid search
        group_kfold2 = GroupKFold(n_splits=n_splits)
        group_kfold2.get_n_splits(X_train, y_train, groups_train[train_index])

        # Normalize data   
        if norm:
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # For the folds train vs test set, cross validate
        #smaller values specify stronger regularization.
        parameters = {'C':[0.00001, 0.0001, 0.01, 0.1, 1, 10]}
        ## Define model ##
        inner_clf = GridSearchCV(
            SVC(kernel="linear", class_weight='balanced'),
            parameters,
            cv=GroupKFold(n_splits=5),
            scoring = 'balanced_accuracy',
            return_train_score=True)
        ## Run the model on the current folds training data - this will run nested cross val ## 
        inner_clf.fit(X_train, y_train, groups_train[train_index])

        #print('train:',inner_clf.predict(X_train))
        print("BEST C:", inner_clf.best_params_['C'])
        print("inner score: ", inner_clf.score(X_train, y_train))
        print(f'inner roc: {roc_auc_score(y_train, inner_clf.predict(X_train))}')
        inner_clf_score = np.hstack((inner_clf_score, inner_clf.score(X_train, y_train)))
        
        
        # Find the best hyperparameter
        C_best_i = inner_clf.best_params_['C']
        C_best.append(C_best_i)

        # Train the classifier with the best hyperparameter using entire train set 
        classifier = SVC(kernel='linear', C=C_best_i, class_weight='balanced')
        clf = classifier.fit(X_train, y_train)
        
        # Test the classifier on held out test dataset 
        score = clf.score(X_test, y_test)
        clf_score = np.hstack((clf_score, score))
        print('test predictions:', classifier.predict(X_test))
        
        roc = roc_auc_score(y_test, classifier.predict(X_test))
        roc_score = np.hstack((roc_score, roc))
        print("Outer score: ", score, 'roc:', roc)

    return np.mean(clf_score), roc_score, np.mean(inner_clf_score), np.mean(roc_score)





def clf_GroupKFold_roc_cross_L2(X1, y1, X2, y2, groups_train, groups_test, n_splits, norm=True):
    """
    We must split data into two different train / test sets because there are different numbers of button presses
    for [SM, SC] and [OM, OC]
    """
    # split into 5 groups # 
    group_kfold = GroupKFold(n_splits)
    group_kfold.get_n_splits(X1, y1, groups_train)
 
    
    ## set vars ## 
    clf_score = np.array([])
    roc_score = np.array([])
    inner_clf_score = np.array([])
    C_best = []

    ## Iterate through each fold ## 
    for cond_set1, cond_set2 in zip(group_kfold.split(X1, y1, groups_train), group_kfold.split(X2, y2, groups_test)):        
        # cond_set 1 is an array of len 2 consisting of train indices, test indices, for the first group 
        train_ind = 0
        test_ind = 1
        train_index = cond_set1[train_ind]
        test_index = cond_set2[test_ind]
        ## split data appropriately into five different train / test splits
        X_train = X1[train_index]
        y_train = y1[train_index]
        X_test = X2[test_index]
        y_test = y2[test_index]
        print(f'x train: {X_train.shape} y_train: {y_train.shape} x_test:{X_test.shape} y_test: {y_test.shape}')
        
        ## convert y's to arrays ## 
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        ## Further split into training groups for grid search
        group_kfold2 = GroupKFold(n_splits=n_splits)
        group_kfold2.get_n_splits(X_train, y_train, groups_train[train_index])

        # Normalize data   
        if norm:
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # For the folds train vs test set, cross validate
        # smaller values specify stronger regularization.
        parameters = {'C':[0.00001, 0.0001, 0.01, 0.1, 1, 10]}
        ## Define model ##
        inner_clf = GridSearchCV(
            LogisticRegression(penalty='l2', class_weight='balanced'),
            parameters,
            cv=GroupKFold(n_splits),
            scoring='balanced_accuracy',
            return_train_score=True)
        ## Run the model on the current folds training data - this will run nested cross val ## 
        inner_clf.fit(X_train, y_train, groups_train[train_index])

        # print('train:',inner_clf.predict(X_train))
        print("BEST C:", inner_clf.best_params_['C'])
        print("inner score: ", inner_clf.score(X_train, y_train))
        print(f'inner roc: {roc_auc_score(y_train, inner_clf.predict(X_train))}')
        inner_clf_score = np.hstack((inner_clf_score, inner_clf.score(X_train, y_train)))
        
        
        # Find the best hyperparameter
        C_best_i = inner_clf.best_params_['C']
        C_best.append(C_best_i)

        # Train the classifier with the best hyperparameter using entire train set 
        classifier = LogisticRegression(penalty='l2', C=C_best_i, class_weight='balanced')
        clf = classifier.fit(X_train, y_train)
        
        # Test the classifier on held out test dataset 
        score = clf.score(X_test, y_test)
        clf_score = np.hstack((clf_score, score))
        print('test predictions:', classifier.predict(X_test))
        
        roc = roc_auc_score(y_test, classifier.predict(X_test))
        roc_score = np.hstack((roc_score, roc))
        print("Outer score: ", score, 'roc:', roc)

    return np.mean(clf_score), clf_score, np.mean(inner_clf_score), np.mean(roc_score)





# # all windows slurm

# In[ ]:


# Top Directory
top_dir = '/jukebox/graziano/coolCatIsaac/ATM/code/analysis/MVPA/final_9-1-23'
act_dir = top_dir + '/activations'
acc_dir = top_dir + '/classification'

# set analysis vars 
date = '2023-07-17'
## activation dic ## 
act_dict_name = f'shaef_roi_activations_earlyWin_spacenorm_{date}'

# accuracy 
acc_dict_name = f'cross_cond_spacenorm_svm_{date}'

#load activation dictionary
activations = np.load(os.path.join(act_dir, '%s.npy') %(act_dict_name ), allow_pickle=True).item()


# In[ ]:


#roi_list = ['dmPFC', 'rTPJ']
roi_list = ['RH_Cont_Par_2', 'LH_Default_PFC_11'] #'RH_Default_Par_3','LH_Default_PFC_11', 'rTPJ_SM-GLM'

### sbatch ## 
window = int(sys.argv[1])

n_splits = 5

cond_list = [['OM', 'OC', 'SM', 'SC'], ['SM','SC','OM','OC']]

# conditions #
#cond_a = 'OM'
#cond_b = 'OC'
#cond_c =  "SM"
#cond_d = "SC"


#window_list = [1,2,3,5,6,7,8]


# In[ ]:

for cond in cond_list:
    cond_a = cond[0]
    cond_b = cond[1]
    cond_c =  cond[2]
    cond_d = cond[3]
    
    for roi in roi_list:
        # # name of current analysis
        analysis_train = '%s_%s_%s_win%s' %(cond_a, cond_b, roi, window)
        analysis_test = '%s_%s_%s_win%s' %(cond_c, cond_d, roi, window)


        X_train = activations[analysis_train]['X']
        y_train = activations[analysis_train]['y']
        X_test = activations[analysis_test]['X']
        y_test = activations[analysis_test]['y']
        groups_train = activations[analysis_train]['groups']
        groups_test = activations[analysis_test]['groups']

        print(X_train.shape, analysis_train)
        print(y_train.shape, analysis_train)
        print(groups_train.shape, analysis_train)

        print(X_test.shape, analysis_test)
        print(y_test.shape, analysis_test)
        print(groups_test.shape, analysis_test)

        ## classify ## 
        OutAcc_avg, RocAcc_array, InAcc_avg, roc_avg = clf_GroupKFold_roc_cross(X_train, y_train, X_test, y_test, groups_train, groups_test, n_splits)

        ## save ## 
        if os.path.isfile(os.path.join(acc_dir, '%s.npy') %(acc_dict_name)):
            #load activation dictionary
            accuracies = np.load(os.path.join(acc_dir, '%s.npy') %(acc_dict_name ), allow_pickle=True).item()
        else:
            accuracies = {}
            print('creating dic')

        d2 = {"OutAcc": OutAcc_avg, "RocAcc_array": RocAcc_array, "InAcc": InAcc_avg, 'roc' : roc_avg}
        # analysis 1
        accuracies[f'{analysis_train}-{analysis_test}'] = d2

        # save again
        np.save(os.path.join(acc_dir, '%s.npy') %(acc_dict_name), accuracies)
print('saved')






