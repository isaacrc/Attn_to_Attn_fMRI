#!/usr/bin/env bash
# PARTICIPANT LEVEL
echo "leggo"
module load pyger/0.11.0
module load afni

# Start Script # 
home='/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/rez_205/individ_data'
# CHANGE ME #

# Dirs #
mask=/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/mask_10r_n22-subs.nii.gz
tem_dir=${home}/temp/
censor_tab=/jukebox/graziano/coolCatIsaac/ATM/data/work/workspace/censor_hm/covar_hm_tab.txt
cd $home

#contrast="0to3vs3to6 0to3vs3to9 0to3vs3to11"
contrast="-6to-4.5vs-4.5to-3.0 -6to-4.5vs3to4.5 -6to-4.5vs-1.5to0.0 -6to-4.5vs0to1.5 -6to-4.5vs1.5to3.0 -6to-4.5vs3to4.5 -6to-4.5vs4.5to6.0"
for cont in $contrast
do
    pref='3dttest_sm-sc-full_contrast_'
    suf='_zmap'
    cats=${pref}${cont}${suf}
    echo $cont
    echo $cats
    #echo "3dttest_afni_full_smsc-${cont}.sh"
done








