#!/usr/bin/env python
# coding: utf-8

# In[20]:


# python resample.py $IMAGE_TO_RESAMPLE $REFERENCE_IMAGE
#!jupyter nbconvert run_perms-2023-05-15-roc.ipynb --to python


# In[1]:


"""
    ~~~ script updates ~~~
copied from 5-11
5-12: cleaned directory structure
copied from 5-13
- implements roc

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

# In[3]:


#from utils import label_lists, find_cond_index, org_bold_data,load_epi_data, load_roi_mask, intersect_mask


# In[4]:


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


# In[5]:


def classify_groups(X,y, groups, norm):
    """
    sub_out is an index of the subject you'd like to leave out
    """
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X, y, groups)
    clf_score = np.array([])
    inner_clf_score = np.array([])
    C_best = []
    #print("GROUPS:", logo.get_n_splits(groups=groups))
    # Train vs Test
    for train_index, test_index in logo.split(X,y, groups):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print(X_train.shape)
        #print(y_train)

        ## Further split into training groups for grid search
        logo2 = LeaveOneGroupOut()
        logo2.get_n_splits(X_train, y_train, groups[train_index])
        #print("GROUPS-2:", logo2.get_n_splits(groups=groups[train_index]))
        # Normalize data   
        if norm:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # For the present train vs test set, cross validate
        parameters = {'C':[0.00001, 0.0001, 0.01, 0.1, 1, 10]}
        inner_clf = GridSearchCV(
            SVC(kernel='linear'),
            parameters,
            cv=logo2.split(X_train, y_train, groups[train_index]),
            return_train_score=True)
        inner_clf.fit(X_train, y_train)
        #print("inner score: ", inner_clf.score(X_train, y_train))
        inner_clf_score = np.hstack((inner_clf_score, inner_clf.score(X_train, y_train)))

        # Find the best hyperparameter
        C_best_i = inner_clf.best_params_['C']
        C_best.append(C_best_i)

        # Train the classifier with the best hyperparameter using training and validation set
        classifier = SVC(kernel="linear", C=C_best_i)
        clf = classifier.fit(X_train, y_train)

        # Test the classifier
        score = clf.score(X_test, y_test)
        clf_score = np.hstack((clf_score, score))
        #print("Outer score: ", score)

    #print ('Inner loop classification accuracy:', np.mean(inner_clf_score))
    #print('best c: ', C_best_i)
    #print ('Overall accuracy: ', np.mean(clf_score))
    return np.mean(clf_score), np.mean(inner_clf_score)
      


# # New classifier functions

# ## Leave one out with SGDClassifier

# In[6]:




# ## GroupKFold classifier, n_splits = 5



# In[8]:


def clf_GroupKFold_roc(X, y, groups, n_splits, norm=False):
    group_kfold = GroupKFold(n_splits=n_splits)
    group_kfold.get_n_splits(X, y, groups)
    clf_score = np.array([])
    roc_score = np.array([])
    inner_clf_score = np.array([])
    C_best = []
    #print("GROUPS:", group_kfold.get_n_splits(groups=groups))

    ## Iterate through each fold ## 
    for i, (train_index, test_index) in enumerate(group_kfold.split(X,y, groups)):
        print(f'\n FOLD {i}')
        ## split into train / test sets 
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(f' Train indices count: {len(np.unique(groups[train_index]))}')
        print(f' Test indices count: {len(np.unique(groups[test_index]))}')
        print(f'percent 1s of Y set {(np.sum(y[test_index])) / len(y[test_index])}')
        
        ## convert y's to arrays ## 
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        ## Further split into training groups for grid search
        group_kfold2 = GroupKFold(n_splits=n_splits)
        group_kfold2.get_n_splits(X_train, y_train, groups[train_index])

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
            SVC(kernel="linear", class_weight = 'balanced'),
            parameters,
            cv=group_kfold2.split(X_train, y_train, groups[train_index]),
            scoring = 'roc_auc',
            return_train_score=True)
        ## Run the model on the current folds training data - this will run nested cross val ## 
        inner_clf.fit(X_train, y_train, groups[train_index])

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


# In[14]:


def single_perm(clf_fxn, X, y, groups, n_splits):  
    print('begin perm')
    #print("Before: ", label_data)
    ## Shuffle TRs + classify
    np.random.shuffle(y)
    ## we only care about roc
    classif_acc_outer, _, _, roc = clf_fxn(X,y, groups, n_splits)
    print(classif_acc_outer, roc)
    print('end perm', end='\n')
    return classif_acc_outer, roc


# # Run Permutations

# In[11]:


# In[17]:


## Slurm ##
num_perm = int(sys.argv[1])
#win_ind = int(sys.argv[2])

random.seed(num_perm)

# directory # 
# Top Directory
top_dir = '/jukebox/graziano/coolCatIsaac/ATM/code/analysis/MVPA/final_9-1-23'
act_dir = top_dir + '/activations'
perm_dir = top_dir +'/permutations'
perm_results = perm_dir + '/perm_results'
acc_dir = top_dir + '/classification'



# set analysis vars 
date = '2023-07-17'
act_dict_name = f'shaef_roi_activations_earlyWin_spacenorm_{date}'
# perms 
perm_roc_name = f'perm_roc_spacenorm_{date}'

#load activation dictionary
activations = np.load(os.path.join(act_dir, '%s.npy') %(act_dict_name ), allow_pickle=True).item()


# Create Condition information
cond_list = np.array([["SM", "OM"]]) # np.array([["SM", "SC"], ["OM", "OC"], ["SM", "OM"]])
window_list = np.array([5,6])
roi_list = ['RH_Cont_Par_2'] # 'LH_Default_PFC_11'
n_splits = 5


# In[18]:


for roi in roi_list:
    for cond_pair in cond_list:
        print(cond_pair)
        cond_a = cond_pair[0]
        cond_b = cond_pair[1]
        for window in window_list:
            # name of current analysis
            analysis = '%s_%s_%s_win%s' %(cond_a, cond_b, roi, window)
            print(analysis)
            X = activations[analysis]['X']
            y = activations[analysis]['y']
            groups = activations[analysis]['groups']
            
            ## Run analysis ##
            permOuter_acc, permOuter_roc = single_perm(clf_GroupKFold_roc, X, y, groups, n_splits)
            
            # Save perm array ## 
            if os.path.isfile(os.path.join(perm_results, '%s_%s.npy') %(perm_roc_name, analysis)):
                print('exists!')
                ## load ## 
                perm_roc_arr = np.load(os.path.join(perm_results, '%s_%s.npy') % (perm_roc_name, analysis), allow_pickle=True)
                ## update 
                perm_roc_arr = np.hstack((perm_roc_arr, permOuter_roc))
                ## save ## 
                np.save(os.path.join(perm_results, '%s_%s.npy') %(perm_roc_name, analysis), perm_roc_arr)
                print('saved')
            else:
                ## create roc 
                perm_roc_arr = np.hstack((np.array([]), permOuter_roc))
                ## save 
                np.save(os.path.join(perm_results, '%s_%s.npy') %(perm_roc_name, analysis), perm_roc_arr)
                print('creating dic')
            print(f'finish {window}')
        print(f'finish {cond_pair}')
    print(f'finish {roi}')
print('finished permutation ', num_perm)
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[40]:



## load array of permuted accs #
#perm_acc_arr = np.load(os.path.join(perm_dir, '%s_%s.npy') %(perm_dict_name, analysis), allow_pickle=True)


# In[21]:


## scoring not equal to roc_auc in neseted cross validation ##

#acc_dict_name = f'classification_acc_n28roc_{date}'

#accuracies = np.load(os.path.join(acc_dir, '%s.npy') %(acc_dict_name ), allow_pickle=True).item()
for anal in accuracies:
    print(f'{anal} is {accuracies[anal]["OutAcc"]} and roc is {accuracies[anal]["roc"]}')


# In[ ]:





# In[ ]:





# In[ ]:




