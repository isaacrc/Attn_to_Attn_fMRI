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
set afni = "/jukebox/graziano/coolCatIsaac/ATM/data/afni/"
# no sub-010 !
#set curList= ( `cat sublist2.txt` )
set curList = ( sub-000 sub-001 sub-003 sub-004 sub-005 sub-006 sub-007 sub-008 sub-009 sub-011 sub-012 sub-013 sub-014 sub-015 sub-016 sub-017 sub-018 sub-019 sub-020 sub-021)
#set curList = ( sub-000 ) 

echo 'hello'
set run_list = ( 01 02 03 04 05 06 07 08 09 10 )

### 6 ### 


############################################################################################
# BEGIN ITERATION
############################################################################################

foreach subjID ( $curList )
cd ${afni}${subjID}/${subjID}.results
foreach run ( $run_list)
    3dAFNItoNIFTI -prefix ${subjID}_${run}_pb05 ${afni}${subjID}/${subjID}.results/pb05.${subjID}.r${run}.scale+tlrc
    echo "Done ${subjID}"

end
end
