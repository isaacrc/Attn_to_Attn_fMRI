#!/usr/bin/env bash
# PARTICIPANT LEVEL
echo "leggo"
module load pyger/0.11.0
module load afni

# Start Script # 
home='/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/rez_205/individ_data'
# CHANGE ME #
pref='3dttest_sm-sc-full_contrast_0to3vs3to6_zmap'
# Dirs #
mask=/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/mask_10r_n22-subs.nii.gz
tem_dir=${home}/temp/
censor_tab=/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/censor_hm/covar_hm_tab.txt
cd $home

# ** BE sure to include label, ie subject001, then the map!
# Start ttest
3dttest++ -mask $mask -prefix $pref \
-setA Differnces sub-000	sub-000_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-001	sub-001_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-002	sub-002_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-003	sub-003_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-004	sub-004_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-005	sub-005_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-006	sub-006_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-007	sub-007_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-008	sub-008_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-009	sub-009_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-010	sub-010_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-011	sub-011_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-012	sub-012_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-013	sub-013_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-014	sub-014_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-015	sub-015_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-016	sub-016_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-017	sub-017_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-018	sub-018_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-019	sub-019_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-020	sub-020_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
sub-021	sub-021_sm-sc-full_contrast_205_0lag_0to3vs3to6_zmap.nii	\
-Clustsim \
-tempdir $tem_dir
#3dAFNItoNIFTI -prefix $pref ${pref}








