#!/bin/tcsh 

############################################################################################
# SCRIPT SUMMARY
#############################################################################################
#
#  Purpose: ~1~
#	This script was written to preprocess rest data. 
#
#  Inputs ~2~
#   - Raw epi run. Only one input file, so if there are multiple runs, concatenate
#   - Subject's anatomical data
#
#  Outputs ~3~
#	The output is the results directory typically produced by afni's preprocessing script.
#	Specifically, a results directory, a script that is produced and called using the
#	parameters set below, and the output of the mentioned script (output.proc.sub)
#
#  Basic Script Outline ~4~
#
#   - GENERAL: specify batch parameters
#   - BEGIN ITERATION: begin looping through subjects and ensuring correct data is available
#	* Here we also decide what to do with previous preprocessing output
#   - SPECIFY MOTION PARAM: set motion parameters for intialization script
#   - SPECIFY PARAM FOR AFNI PROC: Initialization script that generates proc.sub
#   - RUN: Run preprocessing script that you just generated
#   - CHECK RESULTS: Check results of batch
#
############################################################################################
# GENERAL COMMENTS
############################################################################################
#    BE SURE TO
#	1 ~ setup parameters in general setup
#	2 ~ modify any arguments for afni.proc
#    HAVE FUN
#
############################################################################################
# GENERAL SETUP
############################################################################################
### 1 ###
## set parent directory #
set home = "/jukebox/graziano/coolCatIsaac/ATM/data"
###
echo "made it"
### 2 ###
set output_dir = "$home/afni/"



# no sub-010 !
#set curList= ( `cat sublist2.txt` )
set curList = ( sub-001 sub-002 sub-003 sub-004 sub-005 sub-006 sub-007 sub-008 sub-009 sub-011 sub-012 sub-013 sub-014 sub-015 sub-016 sub-017 sub-018 sub-019 sub-020 sub-021)
set curList = ( sub-013 sub-014 sub-015 sub-016 sub-017 sub-018 sub-019 sub-020 sub-021 sub-000 )
set curList = ( sub-017 sub-018 sub-019 sub-020 sub-021 sub-000 )

### 6 ### 
## VARIABLES TO CHANGE BELOW ## 
# anat
# dset
###

############################################################################################
# SCRIPT OPTIONS
############################################################################################
### 1 ###
## Delete previous results directory if it exists? #
set delete_d = TRUE
###

### 2 ###
## Skip processing if results directory already exists ##
set skip = FALSE
###

### 3 ### 
## Shall we archive old preproc directories?  ## 
set archive = FALSE
###

############################################################################################
# BEGIN ITERATION
############################################################################################

foreach subjID ( $curList )
     set subj_indir = "${home}/bids/${subjID}/ses-01/func"
     set subj_outdir = "$output_dir/$subjID"
     echo ${subj_indir}
     set anat = "${home}/bids/${subjID}/ses-01/anat/${subjID}_ses-01_T1w.nii.gz"


  	##### Skip or delete previous results directory #####
	if ( -d $subj_outdir ) then
        cd $subj_outdir
		if ( "$delete_d" == TRUE ) then
            rm -r $subj_outdir

	  	else if ( ${archive} == TRUE ) then
			if (! -d archive ) then
				mkdir archive
			endif
			mv ${subjID} ${subjID}_2
			mv ${subjID}_2 archive/
			echo "results archived"		
	  	else if ( "$skip" == TRUE ) then
	    		echo "I will skip ${subjID} your majesty"
	   		continue
	  	endif
	endif
	#####
    mkdir -p $subj_outdir
    cd $subj_outdir


    ######################################################
    # SPECIFY MOTION AND TR EXCLUSION PREFERENCES
    ######################################################
    set motion_max = .5
    set outlier_max = 0.1

    # echo "Specify a cost function [e.g. lpc, lpa, lpc+zz (default)]"
    #set costfunc = lpc+zz

    # create a run.afni_proc script in this directory
    cat > run.afni_proc << EOF
    echo "$subj_indir/${subjID}_ses-01_task-Attn_run-?_bold"
    ######################################################
    # SPECIFY PARAMETERS FOR AFNI_PROC.PY
    ######################################################
    # afni/2020.10.21
    # run afni_proc.py to create a single subject processing script
    afni_proc.py -subj_id $subjID    \
            -blocks despike tshift align tlrc volreg blur mask scale regress    \
            -copy_anat $anat  \
            -dsets       \
                $subj_indir/${subjID}_ses-01_task-Attn_run-01_bold.nii \
                $subj_indir/${subjID}_ses-01_task-Attn_run-02_bold.nii \
                $subj_indir/${subjID}_ses-01_task-Attn_run-03_bold.nii \
                $subj_indir/${subjID}_ses-01_task-Attn_run-04_bold.nii \
                $subj_indir/${subjID}_ses-01_task-Attn_run-05_bold.nii \
                $subj_indir/${subjID}_ses-01_task-Attn_run-06_bold.nii \
                $subj_indir/${subjID}_ses-01_task-Attn_run-07_bold.nii \
                $subj_indir/${subjID}_ses-01_task-Attn_run-08_bold.nii \
                $subj_indir/${subjID}_ses-01_task-Attn_run-09_bold.nii \
                $subj_indir/${subjID}_ses-01_task-Attn_run-10_bold.nii \
        -volreg_align_to last                                              \
        -volreg_align_e2a                                                   \
        -align_opts_aea -cost lpc+zz                 \
        -epi_strip 3dSkullStrip                                             \
        -tlrc_base MNI152_T1_2009c+tlrc    \
        -tlrc_NL_warp                                                       \
        -volreg_tlrc_warp                                                   \
        -regress_censor_motion $motion_max \
        -regress_censor_outliers $outlier_max \
        -regress_apply_mot_types demean deriv \
        -execute
EOF

######################################################
# Run preprocessing command
######################################################
    tcsh -xef run.afni_proc |& tee output.proc.${subjID} 
    #tcsh run.afni_proc
    echo "hello"

end
