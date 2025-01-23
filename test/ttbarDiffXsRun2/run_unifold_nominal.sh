#!/bin/bash
timestamp=${1:-'latest'}
subcampaigns=${2:-'mc16a mc16d mc16e'}

sample_dir=${DATA_DIR}/ntuplerTT/latest
outdir=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs_MINI382/NominalUniFold/$timestamp

observables='mtt ptt th_pt tl_pt ytt th_y tl_y'

#######
echo
echo "Generate run configs"

for obs in ${observables[@]}; do
    echo $obs
    python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/createRun2Config.py \
        --sample-dir ${sample_dir} \
        --result-dir ${outdir}/$obs \
        --config-name ${outdir}/configs/runCfg_${obs} \
        --subcampaigns $subcampaigns \
        --observables $obs \
        --run-list nominal
done

echo
echo "Run unfolding"
for obs in ${observables[@]}; do
    echo $obs
    python ${SOURCE_DIR}/run_unfold.py ${outdir}/configs/runCfg_${obs}_nominal.json

    echo
    echo "Make histograms"

    result_dir=${outdir}/$obs/nominal

    python ${SOURCE_DIR}/scripts/make_histogramsv2.py ${result_dir} \
        --binning-config ${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json \
        --observables $obs \
        --apply-corrections \
        --include-ibu --compute-metrics -pp -v
done