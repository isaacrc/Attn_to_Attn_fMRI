#!/usr/bin/env bash
# PARTICIPANT LEVEL
echo "leggo"
module load pyger/0.11.0
module load afni

# Start Script # 
home='/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/rez_205/base6win3_1glm'

# Dirs #
mask=/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/mask_10r_n22-subs.nii.gz
tem_dir=${home}/temp/
censor_tab=/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/censor_hm/covar_hm_tab.txt
cd $home
mkdir -p temp

# ** BE sure to include label, ie subject001, then the map!
# Start ttest

#contrast="-6to-4.5vs-4.5to-3.0 -6to-4.5vs3to4.5 -6to-4.5vs-1.5to0.0 -6to-4.5vs0to1.5 -6to-4.5vs1.5to3.0 -6to-4.5vs3to4.5 -6to-4.5vs4.5to6.0"
contrast="win1 win2 win3"
for cont in $contrast
do
    b4='3dttest_sm-sc-full_contrast_'
    suf='_zmap'
    pref=${b4}${cont}${suf}
    echo $cont
    echo $pref
    echo "sub-000_sm-sc-full_contrast_205_0lag_${cont}_zmap.nii"

3dttest++ -mask $mask -prefix $pref \
-setA Differnces sub-000	sub-000_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-001	sub-001_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-002	sub-002_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-003	sub-003_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-004	sub-004_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-005	sub-005_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-006	sub-006_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-007	sub-007_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-008	sub-008_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-009	sub-009_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-010	sub-010_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-011	sub-011_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-012	sub-012_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-013	sub-013_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-014	sub-014_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-015	sub-015_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-016	sub-016_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-017	sub-017_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-018	sub-018_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-019	sub-019_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-020	sub-020_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
sub-021	sub-021_sm-sc-full_contrast_base6win3_1glm_${cont}_zmap.nii	\
-Clustsim \
-tempdir $tem_dir
done
#3dAFNItoNIFTI -prefix $pref ${pref}
# rsync -rvp isaacrc@scotty.princeton.edu:/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/rez_205/individ_data/3dttest_sm-sc-full_contrast_*_zmap+tlrc* .
# cd /Users/isaacchristian/Desktop/Princeton/RESEARCH/META/results/ttests








