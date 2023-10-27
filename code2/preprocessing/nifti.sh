#!/usr/bin/env bash
# PARTICIPANT LEVEL
echo "leggo"
module load pyger/0.11.0
module load afni

# Start Script # 
home='/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/n28_p3_zscr_4fhwm_shaefGM/single_within'
cd $home

## SM vs SC ##
contrast="win1 win2 win3"

## OM vs OC ## 
#contrast="base win1 win2 win3"

for cont in $contrast
do
    b4='3dttest_sc-n28_sinSubtr_within_'
    suf='_zmap'
    pref=${b4}${cont}${suf}
    echo $cont
    echo $pref


3dAFNItoNIFTI -prefix $pref ${pref}+tlrc
done



# rsync -rvp isaacrc@scotty.princeton.edu:/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/n28/3dttest_smsc-full_contrast_p3HM_GREYM_n28win*_zmap+tlrc* .

# rsync -rvp isaacrc@scotty.princeton.edu:/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/n28/3dttest_omoc-no_within_p3HM_GREYM_n28_*_zmap+tlrc* .

#rsync -rvp isaacrc@scotty.princeton.edu:/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/n28/individ_data/3dttest*_zmap+tlrc* .

# rsync -rvp isaacrc@scotty.princeton.edu:/jukebox/graziano/coolCatIsaac/ATM/data/work/results/bpress_GLM/n28/*.NN1_2sided.1D .

# cd /Users/isaacchristian/Desktop/Princeton/RESEARCH/META/results/ttests








