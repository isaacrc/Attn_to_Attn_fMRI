#!/usr/bin/env bash
# PARTICIPANT LEVEL
echo "leggo"
module load pyger/0.11.0
module load afni

# Start Script # 
home='/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/rez_205/individ_data/'
# CHANGE ME #
pref='3dttest_smonly_0-3vs-3-0_zmap'
# Dirs #
mask=/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/mask_10r_n22-subs.nii.gz
tem_dir=${home}/temp/
censor_tab=/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/censor_hm/covar_hm_tab.txt
cd $home

# ** BE sure to include label, ie subject001, then the map!
# Start ttest
3dttest++ -mask $mask -prefix $pref \
-setA Differnces sub-000	sub-000_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-001	sub-001_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-002	sub-002_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-003	sub-003_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-004	sub-004_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-005	sub-005_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-006	sub-006_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-007	sub-007_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-008	sub-008_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-009	sub-009_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-010	sub-010_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-011	sub-011_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-012	sub-012_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-013	sub-013_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-014	sub-014_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-015	sub-015_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-016	sub-016_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-017	sub-017_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-018	sub-018_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-019	sub-019_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-020	sub-020_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
sub-021	sub-021_sm-b4-after_contrast_-3to0vs0to3_zmap.nii	\
-Clustsim \
-tempdir $tem_dir
#3dAFNItoNIFTI -prefix $pref ${pref}








