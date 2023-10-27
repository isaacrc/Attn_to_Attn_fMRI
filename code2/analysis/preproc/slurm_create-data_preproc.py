#!/usr/bin/env python
# coding: utf-8

# In[1]:


#python resample.py $IMAGE_TO_RESAMPLE $REFERENCE_IMAGE


# In[2]:


#jupyter nbconvert --to python slurm_create-data_preproc.ipynb


# In[3]:


import nibabel as nib

from nilearn.input_data import NiftiMasker , MultiNiftiMasker


# In[4]:


import nilearn as nil


# In[5]:


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
from copy import deepcopy


# In[6]:


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

#%autosave 5
#%matplotlib inline
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


# In[7]:


#%matplotlib inline
from brainiak import image, io
from scipy.stats import stats
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from brainiak import image, io


# In[8]:


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


# In[9]:


import pandas as pd


# In[10]:


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


# In[11]:


# Visualize it as an ROI
from nilearn.plotting import plot_roi
#plot_roi(x)


# In[12]:


from nilearn.image import concat_imgs, resample_img, mean_img
from nilearn.plotting import view_img


# In[13]:


from nilearn import datasets, plotting
from nilearn.input_data import NiftiSpheresMasker

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import concat_imgs, resample_img, mean_img,index_img
from nilearn import image
from nilearn import masking
from nilearn.plotting import view_img
from nilearn.image import resample_to_img


# # Functions 

# In[14]:


# Visualize it as an ROI
from nilearn.plotting import plot_roi


# In[15]:


def load_epi_data(sub, ses, task,run, space):
  # Load MRI file
    if space == "MNI":
        epi_in = os.path.join(data_dir, sub, ses, 'func', "%s_%s_task-%s_run-%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" % (sub, ses, task,run))
    elif space == "T1":
        epi_in = os.path.join(data_dir, sub, ses, 'func', "%s_%s_task-%s_run-%s_space-T1w_desc-preproc_bold.nii.gz" % (sub, ses, task,run))
    else:
        print("wrong load epi input. check this function")
    epi_data = nib.load(epi_in)
    print(epi_data.shape)
    print("Loading data from %s" % (epi_in))
    return epi_data

def load_roi_mask(ROI_name, space):
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
def intersect_mask(sub, num_runs,reg, ses="ses-01",task="Attn"):
    # This is based off of 'load_data' function in template
    # Loads all fMRI runs into a matrix #
    """
    reg = T1 or MNI registration?
    norm_type = by Space or by Time? 
    """
    yoz = []
    print("Begin intersecting, you sexy beast")
    for run in range(1, num_runs + 1):
        if sub == "sub-002":
            if run >=7:
                run = run+1
        # Load epi data 
        epi = load_epi_data(sub,ses,task,run,reg)
        # Mask data
        roi_samp = compute_epi_mask(epi) # -- whole brain
        #roi_samp load_roi_mask(ROI_name,reg) # -- mask

        nifti_masker = NiftiMasker(mask_img=roi_samp)
        maskedData = nifti_masker.fit_transform(epi)
        yoz.append(roi_samp)
    #print(concatenated_data)
    epi_data = nil.masking.intersect_masks(yoz)
    print("all done wit da intersextion (lol)")

    return epi_data


# In[ ]:





# In[18]:


def load_fMRI3d(sub, num_runs,reg, norm_type, ses="ses-01",task="Attn"):
    # This is based off of 'load_data' function in template
    # Loads all fMRI runs into a matrix #
    """
    reg = T1 or MNI registration?
    norm_type = by Space or by Time? - default is by space (rows)
    """
    concatenated_data = []
    int_mask = intersect_mask(sub, 1,reg)
    # 
    for run in range(1, num_runs + 1):
        if sub == "sub-002":
            if run >=7:
                run = run+1
        if sub != "sub-010":
            # Load epi data 
            epi = load_epi_data(sub,ses,task,run,reg)
        else:
            # Load epi data 
            print("sub-10, watch out")
            bad_epi = load_epi_data(sub,ses,task,run,reg)
            good_epi = load_epi_data("sub-001",ses,task,run,reg)
            epi = resample_to_img(bad_epi , good_epi, interpolation='nearest')
            int_mask = resample_to_img(int_mask , good_epi, interpolation='nearest')
        # delete first 9 TRs
        epi = index_img(epi,slice(9,209))
        
        # load confounds
        run_conf = np.asarray(pd.read_csv(os.path.join(confounds + sub + "/func/", 
                                                           '%s_ses-01_task-Attn_run-%s_desc-model_timeseries.csv') % (sub, run)))
        print(run_conf.shape)
        # clean image
        # low_pass= .1,
        clean_bold = image.clean_img(epi, standardize = False, confounds = run_conf[9:], high_pass=1/128, 
                                   t_r=1.5, mask_img = int_mask)
        
        
        #Smooth
        clean_bold = image.smooth_img(clean_bold, fwhm=5)
        
        # F*k it mask off -- Load ROI data
        #roi_samp =load_roi_mask(ROI_name,reg)
        # Pull voxels from epi data # *** may need to change this 
        #nifti_masker = NiftiMasker(mask_img=roi_samp)
        #masked_data = nifti_masker.fit_transform(clean_bold)
        
        #append to cat_dat
        concatenated_data.append(clean_bold)
    "FINISHED YAY BEAST"
    return concatenated_data, epi


# In[27]:


def load_fMRI3d(sub, num_runs,reg, norm_type, mask, ses="ses-01",task="Attn"):
    # This is based off of 'load_data' function in template
    # Loads all fMRI runs into a matrix #
    """
    reg = T1 or MNI registration?
    norm_type = by Space or by Time? - default is by space (rows)
    """
    concatenated_data = []
    
    for run in range(1, num_runs + 1):
        if sub == "sub-002":
            if run >=7:
                run = run+1
        if sub != "sub-010":
            # Load epi data 
            epi = load_epi_data(sub,ses,task,run,reg)
        else:
            # Load epi data 
            print("sub-10, watch out")
            bad_epi = load_epi_data(sub,ses,task,run,reg)
            good_epi = load_epi_data("sub-001",ses,task,run,reg)
            epi = resample_to_img(bad_epi , good_epi, interpolation='nearest')
        # delete first 9 TRs ** change CONFOUNDS AS WELL
        epi = index_img(epi,slice(5,209))
        
        # load confounds
        run_conf = np.asarray(pd.read_csv(os.path.join(confounds + sub + "/func/", 
                                                           '%s_ses-01_task-Attn_run-%s_desc-model_timeseries.csv') % (sub, run)))
        print(run_conf.shape)
        # clean image
        # low_pass= .1,
        clean_bold = image.clean_img(epi, standardize = False, confounds = run_conf[5:], high_pass=1/128, 
                                   t_r=1.5, mask_img = mask)
        
        
        #Smooth
        clean_bold = image.smooth_img(clean_bold, fwhm=5)
        
        # F*k it mask off -- Load ROI data
        #roi_samp =load_roi_mask(ROI_name,reg)
        # Pull voxels from epi data # *** may need to change this 
        #nifti_masker = NiftiMasker(mask_img=roi_samp)
        #masked_data = nifti_masker.fit_transform(clean_bold)
        
        #append to cat_dat
        concatenated_data.append(clean_bold)
    "FINISHED YAY BEAST"
    return concatenated_data, epi


# In[ ]:





# # Define Static VARS

# In[17]:


"""
To do:
- log regression
- optimize hyper parameters
- what accuracies hsould i be expecting
- only train and test on two
- bootstrap
- two splits, not four 
- Try averagin datapoints 
- motion
- test on multiple subjects 
- subject specific space 
- implement z score during training
"""


# In[28]:


####
data_dir = "/jukebox/graziano/coolCatIsaac/ATM/data/bids/derivatives/fmriprep/"
rois_dir = "/jukebox/graziano/coolCatIsaac/ATM/data/work/rois/"
behav_p = '/jukebox/graziano/coolCatIsaac/ATM/data/behavioral'
sav_work = "/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/"
confounds = '/jukebox/graziano/coolCatIsaac/ATM/data/bids/derivatives/fmriprep/afni-head_mot/'
workspace = "/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/"
parc_dir = "/jukebox/graziano/coolCatIsaac/ATM/data/work/rois/schaef_par/MNI/"
sav_fcma = '/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/load_fcma'


# # lesss do it

# In[29]:


##### GENERAL ####
sub_list = ["sub-000","sub-001","sub-003","sub-004","sub-005","sub-006","sub-007","sub-008","sub-009",
            "sub-011","sub-012","sub-013","sub-014"]
sub_list = ["sub-000","sub-001","sub-005","sub-006","sub-007","sub-008","sub-009",
            "sub-011","sub-012","sub-013","sub-014"]
sub_list = ["sub-015", "sub-016","sub-017", "sub-018", "sub-019", "sub-020","sub-021", "sub-021"]
sub_list = ["sub-000","sub-001","sub-002","sub-003","sub-004","sub-005","sub-006","sub-007","sub-008","sub-009",
            "sub-010","sub-011","sub-012","sub-013","sub-014","sub-015", "sub-016","sub-017", 
            "sub-018", "sub-019", "sub-020","sub-021"]
"""
sub_list = ["sub-011","sub-012","sub-013","sub-014","sub-015", "sub-016","sub-017", 
            "sub-018", "sub-019", "sub-020","sub-021"]
"""
#sub_list = ["sub-003"]


###### LOADING VARS #######
# Number of runs to load 
num_runs = 10
# Registration ust be either T1 or MNI
reg = "MNI"
# Registration Space # 
norm_type = "space"
# SUFFIX - change me
suffix = "fwm5_mni_hp_nonorm_205"


# In[34]:


# LOAD group mask
mask_file = os.path.join(sav_fcma, 'mask_10r_n22-subs.nii.gz')
for sub in sub_list:
    print("POW, right in the kisser! Begin", sub)
    # Load all subject run data
    concat, raw_epi = load_fMRI3d(sub, num_runs, reg, norm_type, mask_file)
    np.save(sav_work + sub + "_"+ suffix + ".npy", concat, allow_pickle = True)
print("phew, finished. go grab a cup of tea")
            


# In[ ]:




