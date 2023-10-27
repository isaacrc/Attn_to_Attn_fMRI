#!/usr/bin/env python
# coding: utf-8

# In[1]:


#python resample.py $IMAGE_TO_RESAMPLE $REFERENCE_IMAGE
"""
In this script we create any number of windows and find the contrast between
a baseline and each window. differs from previous script in that all windows 
are calculated within one glm, instead of computing one glm separately for
each contrast - one glm is the better method

6-16: 
    glm head mot now includes matrix that includes 19 motion regressors and all censored trs for that run
    new head motion loader function, which isn't explicitly used here but normally loads 19 motion regressors
    variable names changed so that either om or oc can be loaded 
    switched first subtraction to winX-base
    don't run all subs
    load new bpress data
    
7-17
    can select runs for each condition
    t-stat instead of z-score
8-3
    script will now throw out subs if BOTH runs for a conditon are missing
    creates 1 subtraction within files
    creates directory according to a prefix
8-17
    updated run_dic via head_mot, added the head motion variable
    also added a for loop for both conditions
    fixed single within calculation at end of script - now runs only on INCL subs following GLM
10-14
    this is supposed to be my final reviewed script according to the methods section
    parameters changed from 8-17
    - no smoothing
    - high and low pass (no detrend with signal.clean_img)
    - no high pass in GLM 
    - noise model actually is ar1
    - drift order = none
    - applied high and low pass censoring in the fMRI load method
    - all zscore - previous script was ACTUALLY betas. confusing i know
10-17 (recreated from within)
    - adjust windows so that win1 includes the first bpress
    - delete any overlapping button presses
1-4
    - effect size added 
    


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
from nilearn.image import math_img


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
from nilearn.image import concat_imgs, resample_img, mean_img


# In[3]:


from utils import label_lists, find_cond_index, load_epi_data, load_roi_mask,intersect_mask


# In[4]:


# # Functions 

# In[5]:


def fnd_indices(sub,behav_p):
    behav = pd.read_csv(os.path.join(behav_p, '%s_behav_cleaned.csv') % (sub))
    # Define the column in behav to be used for creating labels # 
    label = behav.iloc[:,1]
    # Create an array of labels [1] AND the order in which runs occured [0]#
    sub_ses_labels = label_lists(label, 200)
    ## Find run sequence, extraction condition indexes from behav data ## 
    return find_cond_index(sub_ses_labels[0])
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
    #np.vstack(lab_inx, ["SM", "SC", "OM", "OC"])
    
def load_confounds(cond_list, sub_list,behav_p,confounds):
    """
    args: 
        cond_list: list of conditions (cond_list=np.array(['SM','SC']))
        sub_list: subjects to extract confounds for
        behav_p: path to the behavioral data
        confounds: path to the confound data
    returns:
        nested dictionary in the form of: conf_sub[sub][cond][img_ind]
        where img_index is the first or second run
    """
    # Confound files

    conf_sub = {}
    for sub in sub_list:
        conf_cond = {}
        for cond in cond_list:
            confs = []
            lab_indic = fnd_indices(sub, behav_p)
            confs.append(np.asarray(pd.read_csv(os.path.join(confounds + sub + "/func/", 
                                                             '%s_ses-01_task-Attn_run-%s_desc-model_timeseries.csv') % (sub, lab_indic[cond][0])))[4:,:])
            confs.append(np.asarray(pd.read_csv(os.path.join(confounds + sub + "/func/",
                                                             '%s_ses-01_task-Attn_run-%s_desc-model_timeseries.csv') % (sub, lab_indic[cond][1])))[4:,:])
            conf_cond[cond] = confs
        conf_sub[sub] = conf_cond
    return conf_sub


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



# In[6]:


def create_event_list(sub, bpress, cond, run_dic, base_onset,comp_onset_list,stim_dur):
    """
    this function reads in a condition for each sub
    and returns the corresponding b4 + after events
    
    sub: subject number
    bpress: array of button press onset times
    cond: which condition do you want to create event dataframe fore
    """
    all_tims = []
    events = {}
    # Convert to array #
    bpress_arr = np.asarray(bpress[sub][cond])
    # Select runs to include according to run_dic, append
    all_tims = all_tims + list(bpress_arr[run_dic[sub][cond]])
    # how much to shift from button press onset


    # events will take  the form of a dictionary of Dataframes, one per run! 
    for run in range(len(all_tims)):
        cond_labs = []
        duration = []
        onsets = []
        window_list = []
        # Do base first, then iterate on windows # 
        cond_labs = cond_labs + ['win0']* len(all_tims[run])
        duration = duration + [stim_dur] * len(all_tims[run])
        # onsets 
        onsets_unshft_prs = list(all_tims[run].astype(float))
        onsets = onsets + [y + base_onset for y in onsets_unshft_prs]

        for idx, comp_onset in enumerate(comp_onset_list):
            # Create a list of 'press' for how many presses there were for each run
            # label evaluates to 'win1'
            cond_labs = cond_labs + ['win'+str(idx+1)]* len(all_tims[run])

            # How long does the button event event last? one tr, so 1.5
            duration = duration + [stim_dur] * len(all_tims[run])

            # these are the corresponding onset times
            onsets = onsets +  [y + comp_onset for y in onsets_unshft_prs]
            # how many windows? add to list
            window_list.append('win'+str(idx+1))
        # Define the events object
        events_ = pd.DataFrame(
            {'onset': onsets, 'trial_type': cond_labs, 'duration': duration})
        # remove the rest condition and insert into the dictionary
        events[run] = events_
    return events, window_list


# In[7]:


def create_event_list2(sub, bpress, cond, run_dic, base_onset,comp_onset_list,stim_dur):
    """
    this function reads in a condition for each sub
    and returns the corresponding b4 + after events
    **** this one centers on win1!
    
    sub: subject number
    bpress: array of button press onset times
    cond: which condition do you want to create event dataframe fore
    """
    all_tims = []
    events = {}
    # Convert to array #
    bpress_arr = np.asarray(bpress[sub][cond])
    # Select runs to include according to run_dic, append
    all_tims = all_tims + list(bpress_arr[run_dic[sub][cond]])
    # how much to shift from button press onset


    # events will take  the form of a dictionary of Dataframes, one per run! 
    for run in range(len(all_tims)):
        cond_labs = []
        duration = []
        onsets = []
        window_list = []
        # Do base first, then iterate on windows # 
        cond_labs = cond_labs + ['win1']* len(all_tims[run])
        duration = duration + [stim_dur] * len(all_tims[run])
        # onsets 
        onsets_unshft_prs = list(all_tims[run].astype(float))
        #onsets = onsets + [y + base_onset for y in onsets_unshft_prs]
        # Find the TR which the bpress occured, then find the previous tr to get the first 3s window
        onsets = onsets + [tim - tim % 3  for tim in onsets_unshft_prs]

        for idx, comp_onset in enumerate(comp_onset_list):
            # Create a list of 'press' for how many presses there were for each run
            if idx > 0: idx+=1 ## ** not sure if this will work, TEST**
            # label evaluates to 'win1'
            cond_labs = cond_labs + ['win'+str(idx)]* len(all_tims[run])

            # How long does the button event event last? one tr, so 1.5
            duration = duration + [stim_dur] * len(all_tims[run])

            # these are the corresponding onset times
            onsets = onsets + [tim - tim % 3 + comp_onset  for tim in onsets_unshft_prs]
            # how many windows? add to list
            window_list.append('win'+str(idx))
        # Define the events object
        events_ = pd.DataFrame(
            {'onset': onsets, 'trial_type': cond_labs, 'duration': duration})
        # remove the rest condition and insert into the dictionary
        events[run] = events_
    return events, window_list


# # Define Static VARS

# In[8]:


####
data_dir = "/jukebox/graziano/coolCatIsaac/ATM/data/bids/derivatives/fmriprep/"
rois_dir = "/jukebox/graziano/coolCatIsaac/ATM/data/work/rois/"
behav_p = '/jukebox/graziano/coolCatIsaac/ATM/data/behavioral'
load_bpress = "/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/behav"
load_fmri = "/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/"
out_dir = "/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/"
confounds_dir = '/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/censor_hm/'
afni_dir = '/jukebox/graziano/coolCatIsaac/ATM/data/afni/all_sub_runs/'


# # Sublist

# In[9]:




## OTHER ## 
"""
Excludes sub-000:  no button presses for other_n_monitor.
Script will automatically include runs based on head motion conditions

"""

## SELF 
"""
* not excluding anything even tho sub-009: one button press...? 
"""
sub_list = ["sub-000","sub-001","sub-002","sub-003","sub-004","sub-005","sub-006","sub-007","sub-008","sub-009",
            "sub-010","sub-011","sub-012","sub-013","sub-014","sub-015", "sub-016","sub-017", 
            "sub-018", "sub-019", "sub-020","sub-021",'sub-022','sub-023','sub-024','sub-025','sub-026','sub-027']

"""
Issues:
- sub-000: no button presses for other_n_monitor. values are 1.0 instead of "-1.0". Fucks errthang up **SOLVED
- sub-019: no button presses for other_n_monitor. Values is None, instead of "-1.0". Fucks errthang up **SOLVED
    * only one button press for other count! check effort... may be shit subject
- sub-009 only has one button press for SM **Not an issue for button press counts - just interval calculations

"""


# # Analysis variables

# In[21]:


## Set ewwing ##
# ** Only need to change inputs below!
hm_thresh = str(3)
prefix = 'between'
#prefix1= 'within'
sav_dir = 'n28_p'+hm_thresh+'_betas_4fhwm_hp001p25_shaefGM_excOvlp'

# vars of interest
base_onset = -6
# how many windows and what is the onset start time
# we're now centering on the TR in which the bpress occured - onset of win 1
# -3-3,-3+3,-3+6
comp_onset_list = [-3,0,3]
stim_dur = 3

# subject dics #
group_sub_glm_a = {}
group_sub_glm_b = {}

# conditions #
cond_all = np.asarray([['SM','SC'],['OM', 'OC']])
#cond_all = np.asarray([['OM', 'OC']])

# sublist for individ analysis below # 
excl_sub_list = []

# Create dir
path = out_dir+sav_dir
try: 
    os.mkdir(os.path.join(path))
except OSError as error: 
    print(error)  


# # GLM INFO

# In[12]:


tr = 1.5  # repetition time is 1 second
n_scans = 205 # the acquisition comprises 204 scans
# In seconds what is the lag between a stimulus onset and the peak bold response
sec_lag = 0
frame_times = np.arange(n_scans) * tr  # here are the correspoding frame times given TRs


# In[13]:


## LOAD MASK # 
int_mask = nib.load('/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/load_fcma/mask_10r_n22-subs.nii.gz')
gm_intmask = nib.load('/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/inter_allsubs_.01postresamp_MNI.nii')
gm_shaef = nib.load('/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/shaef_gm_MNI_mask.nii')


# In[14]:


## BPRESS behavioral data ## 
bpress = dict(enumerate(np.load(os.path.join(load_bpress, "n28_4_conds_ts_press.npy"), 
                                allow_pickle=True).flatten(),1))[1]


# In[15]:


## Bpress behavioral data -- overlap removed ##
bpress = dict(enumerate(np.load(os.path.join(load_bpress, "n28_4_conds_ts_press_ovrlpREMOV.npy"), 
                                allow_pickle=True).flatten(),1))[1]


# In[16]:


### runs to exclude with head motion accounted for and missing bpress runs deleted
run_dic = dict(enumerate(np.load(os.path.join(confounds_dir, "n28_runs_2_include_removNoBpress_delHMruns_threshp%s.npy") %(hm_thresh), 
                                allow_pickle=True).flatten(),1))[1]


# In[ ]:





# # Confounds 

# In[17]:


conf_sub = dict(enumerate(np.load(os.path.join(confounds_dir, 'n28_conf+cens_MERGE_removNoBpress_delHMruns_threshp%s_glm.npy')%(hm_thresh), 
                                          allow_pickle = True).flatten(),1))[1]
    
    


# # GLM mult subs

# In[18]:


fmri_glm = FirstLevelModel(t_r=1.5,
                           signal_scaling=False,
                           hrf_model = 'glover',
                           drift_order=None,
                           mask_img = gm_shaef,
                           high_pass=None,
                           drift_model=None,
                           smoothing_fwhm=4,
                           standardize=True,
                           minimize_memory=False)


# In[19]:


fmri_glm.get_params()


# In[20]:





# # Start

# In[ ]:


for cond_list in cond_all:
    cond_a = cond_list[0]
    cond_b = cond_list[1]
    print('cur conds:', cond_list, 'at HM thresh', hm_thresh)
    
    for sub in sub_list:
        # skip sub if both runs for a condition are unavailable (see head_mot)
        if conf_sub[sub][cond_a] == [] or conf_sub[sub][cond_b] == []: continue
        excl_sub_list.append(sub)
        print('design mat for', sub)
        
        sub_conts_a = {}
        sub_conts_b = {}
        # Create events for all four runs
        events_a,window_list = create_event_list(sub, bpress, cond_a,run_dic, base_onset,comp_onset_list,stim_dur)
        # If rest condition, then use button presses from condition A 
        if cond_b == 'RE': 
            events_b,window_list = create_event_list(sub, bpress, cond_a,run_dic,base_onset,comp_onset_list,stim_dur)
        else:
            events_b,window_list = create_event_list(sub, bpress, cond_b,run_dic,base_onset,comp_onset_list,stim_dur)
            
        # Create design matrix - SM
        design_matrices_a = [make_first_level_design_matrix(frame_times, events_a[df], hrf_model = 'glover',
                      drift_model=None, add_regs=conf_sub[sub][cond_a][df]) for df in events_a]
        # Create design matrix - SC (df +1 cuz regs are index 1,2, not 0, 1)
        design_matrices_b = [make_first_level_design_matrix(frame_times, events_b[df], hrf_model = 'glover',
                      drift_model=None, add_regs=conf_sub[sub][cond_b][df]) for df in events_b]

        # FMRI - high pass, low pass filter
        cond_a_runs_temp, cond_b_runs_temp = load_2conds_runs_fmri(sub,behav_p,cond_a,cond_b,run_dic)
        cond_a_runs = [nil.image.clean_img(unclean_img, detrend=False, 
                                        standardize=False, 
                                        low_pass=.25,high_pass=.001,t_r=1.5) for unclean_img in cond_a_runs_temp]
        cond_b_runs = [nil.image.clean_img(unclean_img, detrend=False, 
                                        standardize=False, 
                                        low_pass=.25,high_pass=.001,t_r=1.5) for unclean_img in cond_b_runs_temp]
        

        # FIT GLM per subject 
        print('fit glm')
        window_list = ['win0','win1','win2','win3']
        for win in window_list:
            # group sub glm is now a nested dictionary with three contrasts
            ## COND A ## 
            fmri_glm = fmri_glm.fit(cond_a_runs, design_matrices=design_matrices_a) 
            # Compute three contrasts #
            sub_conts_a[win] = fmri_glm.compute_contrast(
                    win, output_type='effect_size') #effect_size
            # each window saved as a key
            group_sub_glm_a[sub] = sub_conts_a
            ## Save ## 
            np.save(os.path.join(out_dir+sav_dir, '%s_%s.npy') %(cond_a, prefix), group_sub_glm_a)

            ## COND B ## 
            fmri_glm = fmri_glm.fit(cond_b_runs, design_matrices=design_matrices_b) 
            # Compute three contrasts #
            sub_conts_b[win] = fmri_glm.compute_contrast(
                    win, output_type='effect_size') #effect_size

            group_sub_glm_b[sub] = sub_conts_b
            ## Save ## 
            np.save(os.path.join(out_dir+sav_dir, '%s_%s.npy') %(cond_b,prefix), group_sub_glm_b)


            # subtract images
            full_contrast_win = math_img("img1 - img2",img1=group_sub_glm_a[sub][win],img2=group_sub_glm_b[sub][win])

            # save individual full contrast subtracted images for t-test
            nib.save(full_contrast_win, os.path.join(
                out_dir+sav_dir,'%s_%s_%s_%s_zmap.nii') % (sub,cond_a+'-'+cond_b, prefix, win))
            
            print('finish',win)
        print('finish', sub)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




