#!/usr/bin/env bash
# PARTICIPANT LEVEL
echo "leggo"
module load pyger/0.11.0
module load afni

# **Change me** # 
home='/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/n28_p3_betas_4fhwm_hp001p25_shaefGM_excOvlp'

# Dirs #
#mask=/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/mask_10r_n22-subs.nii.gz
#mask=/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/inter_allsubs_.01postresamp_MNI.nii
mask=/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/shaef_gm_MNI_mask.nii

tem_dir=${home}/temp/

cd $home

# Create directories 
mkdir -p temp
mkdir -p clustsim

# ** BE sure to include label, ie subject001, then the map!
# Start ttest

## SM vs SC ## # **Change me** # 
contrast="win1 win2 win3"

## OM vs OC ## 
#contrast="base win1 win2 win3"

for cont in $contrast
do
    # **Change me** # 
    b4='3dttest_omoc_n26_within-between_4fhwm_hp001p25-excOvlp_'
    suf='_zmap'
    pref=${b4}${cont}${suf}
    echo $cont
    echo $pref

3dttest++ -mask $mask -prefix $pref \
-setA Differnces sub-000	sub-000_OM-OC_within_between_${cont}_zmap.nii	\
sub-001	sub-001_OM-OC_within_between_${cont}_zmap.nii	\
sub-002	sub-002_OM-OC_within_between_${cont}_zmap.nii	\
sub-003	sub-003_OM-OC_within_between_${cont}_zmap.nii	\
sub-004	sub-004_OM-OC_within_between_${cont}_zmap.nii	\
sub-005	sub-005_OM-OC_within_between_${cont}_zmap.nii	\
sub-006	sub-006_OM-OC_within_between_${cont}_zmap.nii	\
sub-007	sub-007_OM-OC_within_between_${cont}_zmap.nii	\
sub-008	sub-008_OM-OC_within_between_${cont}_zmap.nii	\
sub-009	sub-009_OM-OC_within_between_${cont}_zmap.nii	\
sub-010	sub-010_OM-OC_within_between_${cont}_zmap.nii	\
sub-011	sub-011_OM-OC_within_between_${cont}_zmap.nii	\
sub-013	sub-013_OM-OC_within_between_${cont}_zmap.nii	\
sub-014	sub-014_OM-OC_within_between_${cont}_zmap.nii	\
sub-015	sub-015_OM-OC_within_between_${cont}_zmap.nii	\
sub-016	sub-016_OM-OC_within_between_${cont}_zmap.nii	\
sub-017	sub-017_OM-OC_within_between_${cont}_zmap.nii	\
sub-018	sub-018_OM-OC_within_between_${cont}_zmap.nii	\
sub-020	sub-020_OM-OC_within_between_${cont}_zmap.nii	\
sub-021	sub-021_OM-OC_within_between_${cont}_zmap.nii	\
sub-022	sub-022_OM-OC_within_between_${cont}_zmap.nii	\
sub-023	sub-023_OM-OC_within_between_${cont}_zmap.nii	\
sub-024	sub-024_OM-OC_within_between_${cont}_zmap.nii	\
sub-025	sub-025_OM-OC_within_between_${cont}_zmap.nii	\
sub-026	sub-026_OM-OC_within_between_${cont}_zmap.nii	\
sub-027	sub-027_OM-OC_within_between_${cont}_zmap.nii	\
-Clustsim \
-tempdir $tem_dir

3dAFNItoNIFTI -prefix $pref ${pref}+tlrc

done

mv *NN* clustsim
mv *CSim* clustsim

# sub-019	sub-019_OM-OC_dubSubtr_between_${cont}_zmap.nii	\
# sub-012	sub-012_OM-OC_dubSubtr_between_${cont}_zmap.nii	\

#3dAFNItoNIFTI -prefix $pref ${pref}
# rsync -rvp isaacrc@scotty.princeton.edu:/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/n28_p3_zscr_NOfhwm_hplp_shaefGM/3dttest_*_zmap+tlrc* .

# rsync -rvp isaacrc@scotty.princeton.edu:/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/n26_p3/3dttest_smsc-p3HM_GREYM_n26_*_zmap+tlrc* .

#rsync -rvp isaacrc@scotty.princeton.edu:/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/n28/individ_data/3dttest*_zmap+tlrc* .

# rsync -rvp isaacrc@scotty.princeton.edu:/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/n28/*.NN1_2sided.1D .

# cd /Users/isaacchristian/Desktop/Princeton/RESEARCH/META/results/ttests








