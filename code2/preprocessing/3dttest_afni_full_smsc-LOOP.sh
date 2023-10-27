#!/usr/bin/env bash
# PARTICIPANT LEVEL
echo "leggo"
module load pyger/0.11.0
module load afni

# Start Script # 
home='/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/n28'

# Dirs #
#mask=/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/mask_10r_n22-subs.nii.gz
#mask=/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/inter_allsubs_.01postresamp_MNI.nii
mask=/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/shaef_gm_MNI_mask.nii

tem_dir=${home}/temp/
censor_tab=/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/censor_hm/covar_hm_tab.txt
cd $home

# Create directories 
mkdir -p temp
mkdir -p clustsim

# ** BE sure to include label, ie subject001, then the map!
# Start ttest

## SM vs SC ##
contrast="base win1 win2 win3"

## OM vs OC ## 
#contrast="base win1 win2 win3"

for cont in $contrast
do
    b4='3dttest_smsc-no_within_p3HM_GREYM_n28_'
    suf='_zmap'
    pref=${b4}${cont}${suf}
    echo $cont
    echo $pref

3dttest++ -mask $mask -prefix $pref \
-setA Differnces sub-000	sub-000_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-001	sub-001_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-002	sub-002_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-003	sub-003_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-004	sub-004_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-005	sub-005_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-006	sub-006_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-007	sub-007_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-008	sub-008_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-009	sub-009_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-010	sub-010_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-011	sub-011_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-012	sub-012_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-013	sub-013_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-014	sub-014_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-015	sub-015_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-016	sub-016_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-017	sub-017_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-018	sub-018_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-019	sub-019_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-020	sub-020_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-021	sub-021_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-022	sub-022_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-023	sub-023_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-024	sub-024_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-025	sub-025_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-026	sub-026_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
sub-027	sub-027_SM-SC-full_contrast_n28_base6win3_1glm_p3HM_no-within_${cont}_zmap.nii	\
-Clustsim \
-tempdir $tem_dir
done

mv *NN* clustsim

#3dAFNItoNIFTI -prefix $pref ${pref}
# rsync -rvp isaacrc@scotty.princeton.edu:/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/n28/3dttest_smsc-full_contrast_p3HM_GREYM_n28win*_zmap+tlrc* .

# rsync -rvp isaacrc@scotty.princeton.edu:/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/n28/3dttest_omoc-no_within_p3HM_GREYM_n28_*_zmap+tlrc* .

#rsync -rvp isaacrc@scotty.princeton.edu:/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/n28/individ_data/3dttest*_zmap+tlrc* .

# rsync -rvp isaacrc@scotty.princeton.edu:/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/n28/*.NN1_2sided.1D .

# cd /Users/isaacchristian/Desktop/Princeton/RESEARCH/META/results/ttests








