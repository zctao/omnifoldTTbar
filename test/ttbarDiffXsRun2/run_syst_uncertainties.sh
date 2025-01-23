#!/bin/bash
timestamp=${1:-'latest'}
subcampaigns=${2:-'mc16a mc16d mc16e'}

sample_dir=${DATA_DIR}/ntuplerTT/latest
outdir=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs_MINI382/Uncertainties/$timestamp

observables='mtt ptt th_pt tl_pt ytt th_y tl_y'
observables_multidim='ptt_vs_mtt th_pt_vs_mtt ytt_abs_vs_mtt ptt_vs_ytt_abs mtt_vs_ytt_abs mtt_vs_ptt_vs_ytt_abs mtt_vs_th_pt_vs_th_y_abs mtt_vs_th_pt_vs_ytt_abs mtt_vs_th_y_abs_vs_ytt_abs'

systematics_filter=''
#'bTagSF_DL1r_70_eigenvars_B1 CategoryReduction_JET_Pileup_RhoTopology'

######
echo
echo "Generate run configs"

python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/run_uncertainties.py \
    -k central ${systematics_filter} \
    -r ${outdir} \
    -v \
    generate \
    --observables ${observables} \
    -e ${subcampaigns} # --config-string '{"match_dR":0.8}'

echo
echo "Run unfolding"

python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/run_uncertainties.py \
    -k central ${systematics_filter} \
    -r ${outdir} \
    -v \
    run

echo
echo "Make histograms"

python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/run_uncertainties.py \
    -k central ${systematics_filter} \
    -r ${outdir} \
    -v \
    histogram \
    --binning-config ${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json \
    --observables ${observables} ${observables_multidim}

#####
#echo
#echo "Evaluate uncertainties"
#python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/run_uncertainties.py \
#    -k ${systematics_filter} \
#    -r ${outdir} \
#    -v \
#    evaluate \
#    -c central

# Instead run 
# test/ttbarDiffXsRun2/run_uncertainties_eval.sh
# and 
# test/ttbarDiffXsRun2/run_uncertainties_replot.sh