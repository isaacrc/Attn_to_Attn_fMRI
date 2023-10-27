#!/usr/bin/env python
# coding: utf-8

# In[19]:


# python resample.py $IMAGE_TO_RESAMPLE $REFERENCE_IMAGE
#!jupyter nbconvert cross-class-run_perms-2023-05-12.ipynb --to python


# In[1]:


"""
    ~~~ script updates ~~~
copied from 5-11
5-12: cleaned directory structure
copied from 5-12
- implemented cross condition SVM in cross-cond script for temporal and cross cond perms
- changed to correct for the wrong classifier alg
5-24
- changed for n28, cross condition given updates from: run_perms-2023-05-15-roc
"""


# In[2]:


import nibabel as nib

from nilearn.input_data import NiftiMasker , MultiNiftiMasker, NiftiLabelsMasker
import nilearn as nil
import numpy as np 
import os
import os.path
import scipy.io
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_epi_mask
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import SGDClassifier
from copy import deepcopy
import warnings
import sys  
import random
# import logging

import deepdish as dd
import numpy as np

import brainiak.eventseg.event
import nibabel as nib
from nilearn.input_data import NiftiMasker

import scipy.io
from scipy import stats
from scipy.stats import norm, zscore, pearsonr
from scipy.signal import gaussian, convolve
from sklearn import decomposition
from sklearn.model_selection import LeaveOneOut, KFold

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import seaborn as sns 


sns.set(style = 'white', context='talk', font_scale=1, rc={"lines.linewidth": 2})

if not sys.warnoptions:
    warnings.simplefilter("ignore")

"""
from utils import sherlock_h5_data

if not os.path.exists(sherlock_h5_data):
    os.makedirs(sherlock_h5_data)
    print('Make dir: ', sherlock_h5_data)
else: 
    print('Data path exists')
    
from utils import sherlock_dir
"""

random.seed(10)

from brainiak import image, io
from scipy.stats import stats
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from brainiak import image, io
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import LeavePGroupsOut
from nilearn.input_data import NiftiMasker
import pandas as pd
# Import machine learning libraries
from nilearn.input_data import NiftiMasker
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy.stats import sem
from copy import deepcopy
from sklearn.metrics import roc_auc_score
import statistics
# Visualize it as an ROI
from nilearn.plotting import plot_roi
#plot_roi(x)
from nilearn.image import concat_imgs, resample_img, mean_img
from nilearn.plotting import view_img
from nilearn import datasets, plotting
from nilearn.input_data import NiftiSpheresMasker

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import concat_imgs, resample_img, mean_img,index_img
from nilearn import image
from nilearn import masking
from nilearn.plotting import view_img
from nilearn.image import resample_to_img
from scipy.spatial.distance import squareform
# Visualize it as an ROI
from nilearn.plotting import plot_roi
import statsmodels.stats.multitest as st
from nilearn import connectome
from nilearn import image
from scipy.spatial.distance import squareform
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import LeaveOneGroupOut
from nilearn import input_data
from nilearn.plotting import plot_glass_brain
from nilearn.masking import apply_mask
import random


# # Functions 

# ## GroupKFold classifier, n_splits = 5

# In[3]:


def clf_GroupKFold_cross(X1, y1, X2, y2, groups_train, groups_test, norm):
    """
    We must split data into two different train / test sets becaz there are diff number of button presses
    for [SM, SC] and [OM, OC]
    """
    # split into 5 groups # 
    group_kfold = GroupKFold(n_splits=5)
    group_kfold.get_n_splits(X1, y1, groups_train)
    ## split dataset two
    group_kfold.get_n_splits(X2, y2, groups_test)
    group_kfold.split(X2,y2, groups_test)
    
    ## Set vars ## 
    clf_score = np.array([])
    inner_clf_score = np.array([])
    C_best = []

    # Train vs Test
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
        
        # convert to arrays #
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        ## Further split into training groups for grid search
        group_kfold2 = GroupKFold()
        group_kfold2.get_n_splits(X_train, y_train, groups_train[train_index])
        print("GROUPS-2:", group_kfold2.get_n_splits(groups=groups_train[train_index]))

        # Normalize data   
        if norm:
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # For the present train vs test set, cross validate
        parameters = {'C':[0.00001, 0.0001, 0.01, 0.1, 1, 10]}
        inner_clf = GridSearchCV(
            SVC(kernel="linear", class_weight='balanced'),
            parameters,
            cv=group_kfold2.split(X_train, y_train, groups_train[train_index]),
            return_train_score=True)
        inner_clf.fit(X_train, y_train)
        print("inner score: ", inner_clf.score(X_train, y_train))
        inner_clf_score = np.hstack((inner_clf_score, inner_clf.score(X_train, y_train)))

        # Find the best hyperparameter
        C_best_i = inner_clf.best_params_['C']
        C_best.append(C_best_i)

        # Train the classifier with the best hyperparameter using training and validation set
        classifier = SVC(kernel='linear', C=C_best_i, class_weight='balanced')
        clf = classifier.fit(X_train, y_train)

        # Test the classifier
        score = clf.score(X_test, y_test)
        clf_score = np.hstack((clf_score, score))
        print("Outer score: ", score)

    return np.mean(clf_score), clf_score, np.mean(inner_clf_score)


# In[4]:


def clf_GroupKFold_roc_cross(X1, y1, X2, y2, groups_train, groups_test, n_splits, norm=True):
    """
    We must split data into two different train / test sets becaz there are diff number of button presses
    for [SM, SC] and [OM, OC]
    """
    # split into 5 groups # 
    group_kfold = GroupKFold(n_splits=5)
    group_kfold.get_n_splits(X1, y1, groups_train)
    ## split dataset two
    group_kfold.get_n_splits(X2, y2, groups_test)
    group_kfold.split(X2,y2, groups_test)
    
    
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
            SVC(kernel="linear"),
            parameters,
            cv=group_kfold2.split(X_train, y_train, groups_train[train_index]),
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

    return np.mean(clf_score), clf_score, np.mean(inner_clf_score), np.mean(roc_score)


# In[5]:


def single_perm(clf_fxn, X_train, y_train, X_test, y_test, groups_train, groups_test, n_splits):
    print('begin perm')
    #print("Before: ", label_data)
    ## Shuffle TRs + classify
    np.random.shuffle(y_train)
    np.random.shuffle(y_test)
    classif_acc_outer, _, _, roc = clf_fxn(X_train, y_train, X_test, y_test, groups_train, groups_test, n_splits)

    print('end perm', end='\n')
    return classif_acc_outer, roc


# # Run Permutations


## Slurm ##
num_perm = int(sys.argv[1])


random.seed(num_perm)



# directory # 
# Top Directory
top_dir = '/jukebox/graziano/coolCatIsaac/ATM/code/analysis/MVPA/final_9-1-23'
act_dir = top_dir + '/activations'
perm_dir = top_dir +'/permutations'
perm_results = perm_dir + '/perm_results'

# set analysis vars 
date = '2023-07-17'
norm = 'space'
act_dict_name = f'shaef_roi_activations_earlyWin_spacenorm_{date}'

# perms 
perm_roc_name = f'cross_cond_perm_{norm}norm_{date}'

#load activation dictionary
activations = np.load(os.path.join(act_dir, '%s.npy') %(act_dict_name ), allow_pickle=True).item()


# In[14]:



### Ipython  checks ###
roi_list = ['LH_Default_PFC_11']#'RH_Cont_Par_2', 
window_list = [4]
n_splits = 5

# conditions #
#cond_a = 'OM'
#cond_b = 'OC'
#cond_c =  "SM"
#cond_d = "SC"
#num_perm = 1 
cond_list = [['OM','OC','SM','SC'], ['SM','SC','OM','OC']]

# In[18]:

for cond in cond_list:
    cond_a = cond[0]
    cond_b = cond[1]
    cond_c = cond[2]
    cond_d = cond[3]
    for roi in roi_list:
        for window in window_list:
            # # name of current analysis
            analysis_train = '%s_%s_%s_win%s' %(cond_a, cond_b, roi, window)
            analysis_test = '%s_%s_%s_win%s' %(cond_c, cond_d, roi, window)
            analysis_lab = analysis_train + "-" + analysis_test

            # load activations # 
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

            ## Run analysis ##
            permOuter_acc, permOuter_roc = single_perm(clf_GroupKFold_roc_cross, X_train, y_train, X_test, y_test, groups_train, groups_test, n_splits)

            # Save perm array ## 
            if os.path.isfile(os.path.join(perm_results, '%s_%s.npy') %(perm_roc_name, analysis_lab)):
                print('exists!')
                ## load ## 
                perm_roc_arr = np.load(os.path.join(perm_results, '%s_%s.npy') % (perm_roc_name, analysis_lab), allow_pickle=True)
                ## update 
                perm_roc_arr = np.hstack((perm_roc_arr, permOuter_roc))
                ## save ## 
                np.save(os.path.join(perm_results, '%s_%s.npy') %(perm_roc_name, analysis_lab), perm_roc_arr)
                print('saved')
            else:
                ## screate roc 
                perm_roc_arr = np.hstack((np.array([]), permOuter_roc))
                ## save 
                np.save(os.path.join(perm_results, '%s_%s.npy') %(perm_roc_name, analysis_lab), perm_roc_arr)
                print('creating dic')
            print(f'finish {window}')
    print(f'finish {roi}')
    print('finished permutation ', num_perm)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


#permOuter_acc, permInner_acc = single_perm(clf_GroupKFold, X, y, groups, True)


# In[39]:


"""
# Save perm array ## 
if os.path.isfile(os.path.join(perm_dir, '%s_%s.npy') %(perm_dict_name, analysis)):
    print('exists!')
    perm_acc_arr = np.load(os.path.join(perm_dir, '%s_%s.npy') % (perm_dict_name, analysis), allow_pickle=True)
    perm_acc_arr = np.hstack((perm_acc_arr, permOuter_avg))
    np.save(os.path.join(perm_dir, '%s_%s.npy') %(perm_dict_name, analysis), perm_acc_arr)
    print('saved')
else:
    perm_acc_arr = np.hstack((np.array([]), permOuter_avg))
    np.save(os.path.join(perm_dir, '%s_%s.npy') %(perm_dict_name, analysis), perm_acc_arr)
    print('creating dic')
    
    
print('finished', num_perm)
"""


# In[34]:



## load array of permuted accs #
#perm_acc_arr = np.load(os.path.join(perm_results, '%s_%s.npy') %(perm_dict_name, analysis_train+"-"+analysis_test), allow_pickle=True)


# In[35


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




