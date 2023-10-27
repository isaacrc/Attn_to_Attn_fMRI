#!/usr/bin/env python
# coding: utf-8

# In[40]:


#python resample.py $IMAGE_TO_RESAMPLE $REFERENCE_IMAGE


# In[41]:


#!jupyter nbconvert bpress_2023-6-1.ipynb --to python


# In[42]:


"""
~~~ Script description ~~~
adapted from troubshoot_bpress_slopes_6-24-22
adapted from ATM_bpress_within-between_1-4-23_rest.py on 1/31/23
- changed bpress to exclude overlap
changed path, added sys.argv for multiple jobs 5/11/23
- roi_activations_2023-05-11.npy is the same dictionary as
  mvpa_results_2023-05-11.npy with more windows and better name  
5/13
- Iterate across subs 
this script specifically looks at 28 subs for dmPFC and rTPJ
5/24
- create function that normalizes over space instead of time
6-1
- ammend for searchlight analysis w 200 schaefer rois
"""


# In[43]:


import nibabel as nib

from nilearn.input_data import NiftiMasker , MultiNiftiMasker

import nilearn as nil
"""
Plotting!
https://nilearn.github.io/plotting/index.html

"""
from nilearn import plotting
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

import pandas as pd
from nilearn.plotting import plot_glass_brain

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

from nilearn import datasets, plotting
from nilearn.input_data import NiftiSpheresMasker

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import concat_imgs, resample_img, mean_img, index_img
from nilearn import image
from nilearn import masking
from nilearn.plotting import view_img
from nilearn.plotting import plot_design_matrix
from nilearn.reporting import get_clusters_table
from nilearn import input_data
from nilearn import datasets


from nilearn.plotting import plot_roi


# In[44]:


from nilearn.image import resample_to_img


# # Functions 

# In[45]:


## Expand / Label TRs
"""
0 = SM
1 = SC
2 = OM
3 = OC
4 = Re
requires list of labels ouputed by psychopy (column 1 - MM_self_title.started, etc.)
returns label list (order is preserved) and TR labels
"""

def label_lists(label, num_tr):
    b = [[]]
    a = []
    for i in label:
        # substring label in psychopy output
        # if the first three characters == M_s, etc, then add correct indext to string
        if i[1:4] == "M_s":
            a.append("SM")
            b.append([0]*num_tr)
        elif i[1:4] == "C_s":
            a.append("SC")
            b.append([1]*num_tr)        
        elif i[1:4] == "M_o":
            a.append("OM")
            b.append([2]*num_tr)
        elif i[1:4] == "C_o":
            a.append("OC")
            b.append([3]*num_tr)     
        else:
            a.append("Re")
            b.append([4]*num_tr)     
    return a, b[1:]
 


# In[46]:


def find_cond_index(sub_ses_labels):
    """
    For the array of ordered run names (i.e.'Re', 'SM',) find the two indexes per condition
    """ 
    lab_inx = []

    a = []
    b = []
    c = []
    d = []
    e = []

    for i in enumerate(sub_ses_labels):
        if i[1] == "SM":
            # append the index according to where it appeared in the array
            a.append(i[0])
        if i[1] == "SC":
            b.append(i[0])
        if i[1] == "OM":
            c.append(i[0])
        if i[1] == "OC":
            d.append(i[0])

    # Create a dictionary where each key contains the appropriate indexes
    lab_indic = {
        'SM' : a,
        'SC' : b,
        'OM' : c,
        'OC' : d,
        'RE' : [0,9]
    }
    return lab_indic 


# In[47]:


def load_epi_data(sub, ses, task,run, space):
  # Load MRI file
    if space == "MNI":
        epi_in = os.path.join(data_dir, sub, ses, 'func', "%s_%s_task-%s_run-%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" % (sub, ses, task,run))
    elif space == "T1":
        epi_in = os.path.join(data_dir, sub, ses, 'func', "%s_%s_task-%s_run-%s_space-T1w_desc-preproc_bold.nii.gz" % (sub, ses, task,run))
    else:
        print("wrong load epi input. check this function")
    epi_data = nib.load(epi_in)
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


# In[48]:


def intersect_mask(sub, num_runs,reg, ses="ses-01",task="Attn"):
    # This is based off of 'load_data' function in template
    # Loads all fMRI runs into a matrix #
    """
    reg = T1 or MNI registration?
    norm_type = by Space or by Time? 
    """
    yoz = []
    print("Begin intersecting, yeehaw")
    for run in range(1, num_runs + 1):
        # Load epi data 
        epi = load_epi_data(sub,ses,task,run,reg)
        # Load ROI data
        roi_samp = compute_epi_mask(epi)
     #   print(roi_samp)
        #nifti_masker = NiftiMasker(mask_img=roi_samp)
        #maskedData = nifti_masker.fit_transform(epi)
        yoz.append(roi_samp)
    #print(concatenated_data)
    epi_data = nil.masking.intersect_masks(yoz)
    print("all done wit da intersextion (lol)")

    return epi_data


# In[49]:


# find indices of where each condition occured in 10 runs for each sub
def fnd_indices(sub,behav_p):
    behav = pd.read_csv(os.path.join(behav_p, '%s_behav_cleaned.csv') % (sub))
    # Define the column in behav to be used for creating labels # 
    label = behav.iloc[:,1]
    # Create an array of labels [1] AND the order in which runs occured [0]#
    sub_ses_labels = label_lists(label, 200)
    ## Find run sequence, extraction condition indexes from behav data ## 
    return find_cond_index(sub_ses_labels[0])


# In[50]:


# organize into dict
def org_bdata_dic(unsort_bdata, run_indexes, cond_a, cond_b): 
    """
    organize two runs for concatenation
    Two runs of cond_a, then two runs of cond_b, in a dik
    
    """
    bold_dict = {}
    a = [unsort_bdata[run_indexes[cond_a][0]], unsort_bdata[run_indexes[cond_a][1]]]
    b = [unsort_bdata[run_indexes[cond_b][0]], unsort_bdata[run_indexes[cond_b][1]]]
    bold_dict[cond_a] = a
    bold_dict[cond_b] = b
    
    print("concatenated", cond_a, cond_b)
    return bold_dict

# organize into list
def org_bdata_list(unsort_bdata, run_indexes, cond_a, cond_b): 
    """
    organize two runs for concatenation
    Two runs of cond_a, then two runs of cond_b
    
    """
    bold_data = []
    a = [unsort_bdata[run_indexes[cond_a][0]], unsort_bdata[run_indexes[cond_a][1]]]
    b = [unsort_bdata[run_indexes[cond_b][0]], unsort_bdata[run_indexes[cond_b][1]]]
    print("concatenated", cond_a, cond_b)
    return a + b

def org_bdata(unsort_bdata, run_indexes, cond_a, cond_b): 
    """
    organize two runs for concatenation
    Two runs of cond_a, then two runs of cond_b
    
    """
    bold_data = []
    a = [unsort_bdata[run_indexes[cond_a][0]], unsort_bdata[run_indexes[cond_a][1]]]
    b = [unsort_bdata[run_indexes[cond_b][0]], unsort_bdata[run_indexes[cond_b][1]]]
    print("returning", cond_a, cond_b)
    return np.asarray(a), np.asarray(b)


# In[51]:


suffix = "_205_noproc.npy"

# load data into a dictionary
def load_smsc_fmri_list(sub,behav_p,cond_a,cond_b,suffix,int_mask):
    cats = list(np.load(load_fmri + sub + suffix, allow_pickle =True))
    #cats = load_fMRI3d(sub, 10, "MNI", 'space', int_mask)
    # Find run labels from behavioral data
    lab_indic = fnd_indices(sub, behav_p)
    # Organize and concatenate bold data
    return org_bdata_list(cats, lab_indic, cond_a, cond_b)

# Load data into dictionary
def load_smsc_fmri_dic(sub,behav_p,cond_a,cond_b,suffix,int_mask):
    cats = list(np.load(load_fmri + sub + suffix, allow_pickle =True))
    #cats = load_fMRI3d(sub, 10, "MNI", 'space', int_mask)
    # Find run labels from behavioral data
    lab_indic = fnd_indices(sub, behav_p)
    # Organize and concatenate bold data
    return org_bdata_dic(cats, lab_indic, cond_a, cond_b)


# # Rest stuff

# In[52]:


def create_event_list_rest(sub, bpress, cond, cond_alt, run_dic, base_onset,comp_onset_list,stim_dur):
    """
    this function reads in a condition for each sub
    and returns the corresponding b4 + after events
    
    sub: subject number
    bpress: array of button press onset times
    cond: which condition do you want to create event dataframe fore
    """
    all_tims = []
    events = {}
    # Convert alternate bpress to array #
    bpress_arr = np.asarray(bpress[sub][cond_alt])
    # Select runs to include according to run_dic, append
    # * include runs of rest 
    all_tims = all_tims + list(bpress_arr[run_dic[sub]['RE']])
    # how much to shift from button press onset
    
    ## create fake bpress for missing data if only one run ## 
    if len(all_tims) <2 and len(run_dic[sub]['RE']) >1:
        all_tims = list(all_tims[0], all_tims[0])
        
    return all_tims


# In[53]:


def load_2conds_runs_fmri(sub,behav_p,cond_a,cond_b, run_dic, suffix="_205_noproc.npy"):
    """
    read in target conditions + subject info
    output: 2 runs of condition A, as they were presented, then two runs of condition B
    """
    cats = list(np.load(load_fmri + sub + suffix, allow_pickle =True)) 
    # Find run labels from behavioral data
    lab_indic = fnd_indices(sub, behav_p)
    # Organize and concatenate bold data
    a, b = org_bdata(cats, lab_indic, cond_a, cond_b)
    # Get the indices of the runs to include #
    return list(a[run_dic[sub][cond_a]]), list(b[run_dic[sub][cond_b]])


# In[54]:


def load_1cond_runs_fmri(sub,behav_p,cond_a, cond_b, run_dic, targ_run, suffix="_205_noproc.npy"):
    """
    read in target conditions + subject info
    output: 2 runs of condition A, as they were presented, then two runs of condition B
    """
    cats = list(np.load(load_fmri + sub + suffix, allow_pickle =True))
    # Find run labels from behavioral data
    lab_indic = fnd_indices(sub, behav_p)
    # Organize and concatenate bold data
    a, b = org_bdata(cats, lab_indic, cond_a, cond_b)
    # Get the indices of the runs to include #
    return list(a[run_dic[sub][cond_a]])


# In[ ]:





# # Define Static VARS

# In[55]:


####
data_dir = "/jukebox/graziano/coolCatIsaac/ATM/data/bids/derivatives/fmriprep/"
#rois_dir = "/jukebox/graziano/coolCatIsaac/ATM/data/work/rois/"

behav_p = '/jukebox/graziano/coolCatIsaac/ATM/data/behavioral'
load_bpress = "/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/behav"
load_fmri = "/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/"
confounds = '/jukebox/graziano/coolCatIsaac/ATM/data/bids/derivatives/fmriprep/afni-head_mot/'
confounds_dir = '/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/censor_hm/'
## ** change me ** ## 
rois_dir = "/jukebox/graziano/coolCatIsaac/ATM/data/work/rois/results_masks/masks_mvpa"
#rois_dir = "/jukebox/graziano/coolCatIsaac/ATM/data/work/rois"

# load whole brain mask
int_mask = nib.load('/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/load_fcma/mask_10r_n22-subs.nii.gz')


# # dynamic vars

# In[56]:


## sublist for rest ##
"""
no 5,8, 19, 26, 12
sub-005
 - 
sub-008

sub-019
- 
sub-012
- index range issue -- definitely due to the NO selection of runs for fMRI data -- we
select runs to be included for bpress data but not fMRI data -- need to edit
"""

### Rest Subs ### 
sub_list = ["sub-000", "sub-001","sub-002","sub-003","sub-004","sub-006","sub-007",
            "sub-009","sub-010","sub-011","sub-013","sub-014","sub-015", "sub-016","sub-017", 
            "sub-018","sub-020", "sub-021", "sub-022","sub-023","sub-024","sub-025","sub-027"]
## bad subs that do not work for rest ## 
sub_list = ["sub-000", "sub-001","sub-002","sub-003","sub-004",'sub-005', "sub-006","sub-007",'sub-008',
            "sub-009","sub-010","sub-011",'sub-012',"sub-013","sub-014","sub-015", "sub-016","sub-017", 
            "sub-018",'sub-019',"sub-020", "sub-021", "sub-022","sub-023","sub-024","sub-025", "sub-026","sub-027"]


## What threshold of head motion to extract ## 
hm_thresh = str(3)

## cluster coordinates to be extracted 
# coords = [(51,-42,44)]


# In[57]:


len(sub_list)


# # bpress data

# In[58]:


## Bpress behavioral data -- overlap removed ##
bpress = dict(enumerate(np.load(os.path.join(load_bpress, "n28_4_conds_ts_press_ovrlpREMOV.npy"), 
                                allow_pickle=True).flatten(),1))[1]


# In[59]:


### runs to exclude with head motion accounted for and missing bpress runs deleted
run_dic = dict(enumerate(np.load(os.path.join(confounds_dir, "n28_runs_2_include_removNoBpress_delHMruns_threshp%s.npy") %(hm_thresh), 
                                allow_pickle=True).flatten(),1))[1]


# # Confounds 

# In[60]:


conf_sub = dict(enumerate(np.load(os.path.join(confounds_dir, 'n28_conf+cens_MERGE_removNoBpress_delHMruns_threshp%s_glm.npy')%(hm_thresh), 
                                          allow_pickle = True).flatten(),1))[1]


# # Create rest button presses 

# In[61]:



"""
for sub in sub_list:
    print(sub)
    temp = []
    events_b = create_event_list_rest(sub, bpress, "RE",cond_a,run_dic,base_onset,comp_onset_list,stim_dur)
    for runs in range(len(run_dic[sub]['RE'])):
        temp.append(events_b[runs])   
    bpress.setdefault(sub, {}).setdefault('RE',temp)
"""


# # ROI Info

# In[62]:


"""
# sublist is set up top
# sub_list = ['sub-001']

# load in target ROI
# roi = ["l_prim_motor_sm_win2_mask"]
# roi = ["dmPFC_ovlp_mask"]
# roi = ["rTPJ_mask"]

roi = ["l_prim_motor_sm_win2_mask"]
tr = 1.5
high_pass = 1/128
roi_mask = load_roi_mask(roi[0], "MNI")

# REGRESS CONFOUNDS 
##** standardize = False ??? # 


masker = NiftiMasker(mask_img=roi_mask,smoothing_fwhm=2,
                     standardize=True, detrend=True, high_pass=high_pass,
                    t_r=tr)
"""


# # Define main function

# In[63]:


def get_activations(sub_list, roi, roi_voxels, tr_range, cond_list):
    """
    Purpose:
        - extract activations from each ROI at each window, average two TRs within window
    Inputs:
        - sub_list: list of subs to iterate through
        - roi: which roi to extract activations 
        - roi_voxels: number of voxels for the selected ROI
        - tr_range: how many TRs should be averaged over?
        - cond_list: Which two conditions are we comparing
    Outputs:
        - X: Activations for every button press across subs for a given window 
        - Y: array of integers describing if button press is cond a (0) or cond b (1)
        - Groups: array of integers denoting button presses for each sub
    """
    # load in target ROI
    roi = roi
    high_pass = 1/128
    roi_mask = load_roi_mask(roi[0], "MNI")

    # Create masker object and REGRESS CONFOUNDS 
    masker = NiftiMasker(mask_img=roi_mask,smoothing_fwhm=2,
                         standardize=True, detrend=True, high_pass=high_pass,t_r=1.5)

    ## X variable corresponds to a matrix that is: roi voxels x 
    X = np.empty((0, roi_voxels))
    ## Set subject array ## 
    group_id = 0
    groups = np.array([])
    ## set array for labels ## 
    y = np.array([])

    for sub in sub_list:
        ## Which conditions? ## 
        cond_a = cond_list[0]
        cond_b = cond_list[1]

        print(sub)
        # LOAD RUNS FOR COND A AND COND B
        fmri_imgs = load_smsc_fmri_dic(sub,behav_p,cond_a,cond_b,suffix,int_mask)
        # for each condition, extract patterns for each TR and average
        for sing_cond in cond_list:
            # for each run in available run dictionairy
            for run in range(len(run_dic[sub][sing_cond])):
                temp_run = []
                print("condition: ", sing_cond)
                #temp_pred = []
                print("run", run)

                # Fit the masker object to extract a 2d matrix: voxels x TR [454, 205]
                print("sub: ", sub)
                roi_act = masker.fit_transform(fmri_imgs[sing_cond][run], confounds = conf_sub[sub][sing_cond][run])
                print(roi_act.shape)


                # extract BPRESS behavioral data # 
                linez = bpress[sub][sing_cond][run]
                print("button press TRS:", linez)

                # Average activations 
                avg_activations = []
                
                # For each TR #
                for tr in linez:
                    # Find the tr that each onset occured - convert from seconds to TR
                    print(tr)
                    tr = round(tr/1.5)
                    print(tr)
                    temp = []
                    for i in tr_range:
                        try:
                            # IF tr exists in the range of TRs add to temp array before averaging in the next step
                            temp.append(roi_act[tr+i])
                        except:
                            continue
                    print("temp length: ", len(temp))
                    # If TRs exist for this button press, add it to the array # 
                    if (len(temp) > 0):
                        X = np.vstack((X, np.mean(temp, axis=0)))
                        print("X: ", X.shape)
                        groups = np.append(groups, group_id)
                        # note that I changed this to cond_a, cond_b
                        if (sing_cond == cond_a):
                            y = np.append(y, 0)

                        elif (sing_cond == cond_b):
                            y = np.append(y, 1)


        group_id = group_id + 1
    
    return X, y, groups


# In[64]:


def get_activations_spacenorm(sub_list, roi, roi_voxels, tr_range, cond_list):
    """
    Purpose:
        - extract activations from each ROI at each window, average two TRs within window
        - ** only difference from above is we changed standardize to false **
    Inputs:
        - sub_list: list of subs to iterate through
        - roi: which roi to extract activations 
        - roi_voxels: number of voxels for the selected ROI
        - tr_range: how many TRs should be averaged over?
        - cond_list: Which two conditions are we comparing
    Outputs:
        - X: Activations for every button press across subs for a given window 
        - Y: array of integers describing if button press is cond a (0) or cond b (1)
        - Groups: array of integers denoting button presses for each sub
    """
    # load in target ROI
    high_pass = 1/128
    roi_mask = roi

    # Create masker object and REGRESS CONFOUNDS 
    masker = NiftiMasker(mask_img=roi_mask,smoothing_fwhm=2,
                         standardize=False, detrend=True, high_pass=high_pass,t_r=1.5)

    ## X variable corresponds to a matrix that is: roi voxels x 
    X = np.empty((0, roi_voxels))
    ## Set subject array ## 
    group_id = 0
    groups = np.array([])
    ## set array for labels ## 
    y = np.array([])

    for sub in sub_list:
        ## Which conditions? ## 
        cond_a = cond_list[0]
        cond_b = cond_list[1]

        print(sub)
        # LOAD RUNS FOR COND A AND COND B
        fmri_imgs = load_smsc_fmri_dic(sub,behav_p,cond_a,cond_b,suffix,int_mask)
        # for each condition, extract patterns for each TR and average
        for sing_cond in cond_list:
            # for each run in available run dictionairy
            for run in range(len(run_dic[sub][sing_cond])):
                temp_run = []
                print("condition: ", sing_cond)
                #temp_pred = []
                print("run", run)

                # Fit the masker object to extract a 2d matrix: voxels x TR [454, 205]
                print("sub: ", sub)
                roi_act = masker.fit_transform(fmri_imgs[sing_cond][run], confounds = conf_sub[sub][sing_cond][run])
                print(roi_act.shape)


                # extract BPRESS behavioral data # 
                linez = bpress[sub][sing_cond][run]
                print("button press TRS:", linez)

                # Average activations 
                avg_activations = []
                
                # For each TR #
                for tr in linez:
                    # Find the tr that each onset occured - convert from seconds to TR
                    print(tr)
                    tr = int(round(tr/1.5))
                    print(tr)
                    temp = []
                    for i in tr_range:
                        try:
                            # IF tr exists in the range of TRs add to temp array before averaging in the next step
                            temp.append(roi_act[tr+i])
                        except:
                            continue
                    print("temp length: ", len(temp))
                    # If TRs exist for this button press, add it to the array # 
                    if (len(temp) > 0):
                        X = np.vstack((X, np.mean(temp, axis=0)))
                        print("X: ", X.shape)
                        groups = np.append(groups, group_id)
                        # note that I changed this to cond_a, cond_b
                        if (sing_cond == cond_a):
                            y = np.append(y, 0)

                        elif (sing_cond == cond_b):
                            y = np.append(y, 1)


        group_id = group_id + 1
    
    return X, y, groups


# # begin

# In[65]:


"""
DATA:
 - ALL_roi_activations_n28_spacenorm_2023-06-1.npy: This is activation for ALL ROIS from the schaefer atlas,copied from
     the *all* roi activations located in the searchlight folder 
 - roi_activations_n28_2023-05-14.npy : i think this is temporal norm but don't quote me
 - roi_activations_n28_spacenorm_2023-05-24.npy: spatial normalized data for two (maybe three? ) target ROIs. 
     does NOT account for the window shift [in old analysis we were off by one]
 - roi_activations_newWin_n28_spacenorm_2023-05-24.npy: *used in FINAL ANALYSIS* spatial normed with the following 
     window sizes: tr_range_dict = {'1' : np.array([-3,-4]), '2' : np.array([-1,-2]), '3' : np.array([0,1]), 
                 '4' : np.array([2,3]), '5' : np.array([4,5]), '6' : np.array([6,7]), 
                '7' : np.array([8,9]), '8' : np.array([10,11])}
  - ^^^ not sure what used in final analysis means. According to the 'final' directory, these were the window sizes used for the LATE window:
  tr_range_dict = {'1' : np.array([-2,-3]), '2' : np.array([0,-1]), '3' : np.array([1,2]), 
                 '4' : np.array([3,4]), '5' : np.array([5,6]), '6' : np.array([7,8]), 
                '7' : np.array([9,10]), '8' : np.array([11,12])}
    - EARLY window: 
    

"""


# In[66]:


# Top Directory
top_dir = '/jukebox/graziano/coolCatIsaac/ATM/code/analysis/MVPA/final_9-1-23'
act_dir = top_dir + '/activations'
perm_dir = top_dir +'/permutations'


perm_results = perm_dir + '/perm_results'
parc_dir = "/jukebox/graziano/coolCatIsaac/ATM/data/work/rois/schaef_par/MNI/"
load_work = "/jukebox/graziano/coolCatIsaac/ATM/data/work/results/corr_data/"


# In[67]:


### Set RoI ## 
#roi_dict = {'motor': ["l_prim_motor_sm_win2_mask"], 'dmPFC': ["dmPFC_ovlp_mask"], 'rTPJ': ["rTPJ_mask"]}

cond_list = [['SM', 'SC'], ['OM', 'OC']]
#cond_list = [['SM', 'OM']]

## used in searchligh analysis -- i think? but i'm really not sure... -- this is EARLY window (earlyWin)
tr_range_dict = {'1' : np.array([-3,-4]), '2' : np.array([-1,-2]), '3' : np.array([0,1]), 
                 '4' : np.array([2,3]), '5' : np.array([4,5]), '6' : np.array([6,7]), 
                '7' : np.array([8,9]), '8' : np.array([10,11])}

"""
## used in 'final' folder analysis ##  -- this is lateWin
tr_range_dict = {'1' : np.array([-2,-3]), '2' : np.array([0,-1]), '3' : np.array([1,2]), 
                 '4' : np.array([3,4]), '5' : np.array([5,6]), '6' : np.array([7,8]), 
                '7' : np.array([9,10]), '8' : np.array([11,12])}
"""


### Save Vars ## 
date = '2023-07-17'
## activation dic ## 
act_dict_name = f'shaef_roi_activations_earlyWin_spacenorm_{date}'

#perm_dict_name = f'perm_results_{date}_slurmy'

#load activation dictionary
#activations = np.load(os.path.join(act_dir, '%s.npy') %(act_dict_name ), allow_pickle=True).item()


# ## get current atlas info

# In[68]:


dataset = datasets.fetch_atlas_schaefer_2018(n_rois=200)
atlas_filename = dataset.maps
labels = dataset.labels

print(f"Atlas ROIs are located in nifti image (4D) at: {atlas_filename}")


# In[69]:


# set path to atlas #
schaef = "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
#schaef = "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"


# In[70]:


## which ROI?
roi_ind = int(sys.argv[1])
roi_nums = [92, 165, 100]
roi_num_one = roi_nums[roi_ind]

# # Load in and resample ROI

# In[71]:


# Load  sample data for resampling
concat = list(np.load(load_work + 'sub-007' +"_fwm7_mni_norm.npy", allow_pickle = True))
resamp_run = concat[0]
# Load parcellation
#d = nib.load(os.path.join(parc_dir + "Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"))
d = nib.load(os.path.join(parc_dir + schaef))
d_resamp = resample_to_img(d, resamp_run, interpolation='nearest')
# Get parcellation fdata
aparc = d_resamp.get_fdata()
# paracellations scheme
print(f'count parc:{len(np.unique(d_resamp.get_fdata()))}')
print("shape of d object", d_resamp.shape)


# In[72]:


# View ROI # 

roi_num = roi_num_one
print("Begin Parc", roi_num)
# How many voxels for this ROI? #
num_voxels = np.sum(aparc == roi_num)
print("num voxels in one parcel:", num_voxels)
# Create an empty that is the shape of d
roi_tem = np.zeros(d_resamp.shape)
# set all cases where parcel is equal to roi_num, equal to one, everything else zero (creates a mask)
roi_tem[aparc == roi_num] = 1
# Create a nift image of the mask
roi_img = nib.Nifti1Image(roi_tem, affine = d_resamp.affine, header = d_resamp.header)
### set ROI_name #
roi_name = labels[roi_num][10:].decode("utf-8")
#nib.save(roi_img, rois_dir+'/'+ roi_name,)

#plot_roi(roi_img)


# # run analysis

# In[ ]:



print("Begin Parc", roi_num)
# How many voxels for this ROI? #
num_voxels = np.sum(aparc == roi_num)
print("num voxels in one parcel:", num_voxels)
# Create an empty that is the shape of d
roi_tem = np.zeros(d_resamp.shape)
# set all cases where parcel is equal to roi_num, equal to one, everything else zero (creates a mask)
roi_tem[aparc == roi_num] = 1
# Create a nift image of the mask
roi_img = nib.Nifti1Image(roi_tem, affine = d_resamp.affine, header = d_resamp.header)
### set ROI_name #
roi_name = labels[roi_num][10:].decode("utf-8")
print(roi_name)


########### **** Do TPJ SM GLM **** ####### 
if roi_num_one == 100:
    print('doin GLM roi!')
    roi_name = 'rTPJ_SM-GLM'
    roi_img = load_roi_mask('SM_win1-win0_mask', 'MNI')
    num_voxels = np.sum(roi_img.get_fdata() == 1) ## 400ish voxels


for conds in cond_list:
        print(conds)
        for window in tr_range_dict:
            print(f'window {window}')
            # Get activations ---- norm by space or time ? --- #
            X, y, g = get_activations_spacenorm(sub_list, roi_img, roi_voxels = num_voxels, 
                                      tr_range = tr_range_dict[window], cond_list = conds)
            ##### NORMALIZE ACROSS VOX ####
            X = zscore(X, axis = 1)
            ###
            print(X.shape)
            print(y.shape)
            print(g.shape)

            print('now save')
            # set analysis # 
            analysis = '%s_%s_%s_win%s' %(conds[0], conds[1], roi_name, window)
            # Save act dic ## 
            if os.path.isfile(os.path.join(act_dir, '%s.npy') %(act_dict_name)):
                print('exists!')
                activations = np.load(os.path.join(act_dir, '%s.npy') %(act_dict_name ), allow_pickle=True).item()
            else:
                activations = {}
                print('creating dic')
            d2 = {"X": X, "y": y, "groups":g}
            activations[analysis] = d2
            np.save(os.path.join(act_dir, '%s.npy') %(act_dict_name), activations)

            
        


# In[73]:


activations = np.load(os.path.join(act_dir, '%s.npy') %(act_dict_name ), allow_pickle=True).item()
for key in sorted(activations):
    print(key)


# In[79]:


#roi_list = ['RH_Cont_Par_2', 'LH_Default_PFC_11','RH_Default_Par_3']
#roi_list = ['LH_Default_PFC_11',]





# In[ ]:




