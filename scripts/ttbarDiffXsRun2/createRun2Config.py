#!/usr/bin/env python3
import os
import glob
import util

# systematics dictionary
from ttbarDiffXsRun2.systematics import syst_dict

def getSamples_detNP(
    sample_dir, # top directory to read sample files
    category, # "ejets" or "mjets" or "ljets"
    systematics = 'nominal', # type of systematics
    subcampaigns = ["mc16a", "mc16d", "mc16e"]
    ):

    years = []
    for e in subcampaigns:
        if e == "mc16a":
            years += [2015, 2016]
        elif e == "mc16d":
            years += [2017]
        elif e == "mc16e":
            years += [2018]
        else:
            raise RuntimeError(f"Unknown MC subcampaign {e}")

    if category == "ljets":
        channels = ["ljets"] #["ejets", "mjets"]
    elif category == "ejets" or category == "mjets":
        channels = [category]

    ###
    # observed data
    data = [os.path.join(sample_dir, f"obs/{y}/data_0_pseudotop_{c}.root") for c in channels for y in years]

    ###
    # background
    backgrounds = []

    # Fakes
    backgrounds += [os.path.join(sample_dir, f"fakes/{y}/data_0_pseudotop_{c}.root") for c in channels for y in years]

    # W+jets
    backgrounds += [os.path.join(sample_dir, f"systCRL/Wjets_nominal/{e}/Wjets_0_pseudotop_{c}.root") for c in channels for e in subcampaigns]

    # Z+jets
    backgrounds += [os.path.join(sample_dir, f"systCRL/Zjets_nominal/{e}/Zjets_0_pseudotop_{c}.root") for c in channels for e in subcampaigns]

    # other samples
    for bkg in ['singleTop', 'ttH', 'ttV', 'VV']:
        backgrounds += [os.path.join(sample_dir, f"detNP/{bkg}_{systematics}/{e}/{bkg}_0_pseudotop_{c}.root") for c in channels for e in subcampaigns]

    ###
    # signal
    signal = []
    for e in subcampaigns:
        for c in channels:
            s = glob.glob(os.path.join(sample_dir, f"detNP/ttbar_{systematics}/{e}/ttbar_*_pseudotop_parton_{c}.root"))
            s.sort()
            signal += s

    assert(data)
    assert(signal)
    assert(backgrounds)

    return data, signal, backgrounds

def createRun2Config(
        sample_local_dir,
        category, # "ejets" or "mjets" or "ljets"
        outname_config = 'runConfig',
        output_top_dir = '.',
        subcampaigns = ["mc16a", "mc16d", "mc16e"],
        do_bootstrap = False,
        systematics = [],
        common_cfg = {}
    ):

    # get the real paths of the sample directory and output directory
    sample_local_dir = os.path.expanduser(sample_local_dir)
    sample_local_dir = os.path.realpath(sample_local_dir)

    output_top_dir = os.path.expanduser(output_top_dir)
    output_top_dir = os.path.realpath(output_top_dir)

    # in case outname_config comes with an extension
    outname_config = os.path.splitext(outname_config)[0]

    # create the output directory in case it does not exist
    outputdir = os.path.dirname(outname_config)
    if not os.path.isdir(outputdir):
        print(f"Create directory {outputdir}")
        os.makedirs(outputdir)

    # nominal input files
    print("nominal")
    obs_nominal, sig_nominal, bkg_nominal = getSamples_detNP(
        sample_local_dir,
        category = category,
        systematics = 'nominal',
        subcampaigns = subcampaigns
    )

    outdir_nominal = os.path.join(output_top_dir, "nominal", f"output_run2_{category}")

    nominal_cfg = common_cfg.copy()
    nominal_cfg.update({
        "data": obs_nominal,
        "signal": sig_nominal,
        "background": bkg_nominal,
        "outputdir": outdir_nominal,
        "plot_verbosity": 2,
        "run_ibu": False
        })

    # write nominal run configuration
    outname_config_nominal = f"{outname_config}_nominal.json"
    util.write_dict_to_json(nominal_cfg, outname_config_nominal)

    # bootstrap for statistical uncertainties
    if do_bootstrap:
        nresamples = 10
        outdir_resample = os.path.join(output_top_dir, "nominal", f"output_run2_bootstrap_{category}")
        outdir_resample_dict = {
            f"resample{n}" : outdir_resample for n in range(nresamples)
            }

        resample_cfg = common_cfg.copy()
        resample_cfg.update({
            "data": obs_nominal,
            "signal": sig_nominal,
            "background": bkg_nominal,
            "outputdir": outdir_resample_dict,
            "resample_data": True
            })

        # write bootstrap run configuration
        outname_config_bootstrap = f"{outname_config}_bootstrap.json"
        util.write_dict_to_json(resample_cfg, outname_config_bootstrap)

    if not systematics: # no systematic uncertainties to evaluate
        return

    # for systematic uncertainties
    cfg_dict_list = []

    # A special case in which all available systematic uncertainties are included
    include_all_syst = systematics == ["all"]

    for k in syst_dict:
        prefix = syst_dict[k]["prefix"]
        uncertainties = syst_dict[k].get("uncertainties", [])
        for s in uncertainties:

            if not include_all_syst and not f"{prefix}_{s}" in systematics:
                # skip this one
                continue

            for v in syst_dict[k]["variations"]:
                syst = f"{prefix}_{s}_{v}"

                print(syst)

                obs_syst, sig_syst, bkg_syst = getSamples_detNP(
                    sample_local_dir,
                    category = category,
                    systematics = syst,
                    subcampaigns = subcampaigns
                )

                outdir_syst = os.path.join(output_top_dir, syst, f"output_run2_{category}")

                syst_cfg = common_cfg.copy()
                syst_cfg.update({
                    "data": obs_syst,
                    "signal": sig_syst,
                    "background": bkg_syst,
                    "outputdir": outdir_syst,
                    "load_models": outdir_nominal,
                    "nruns": 10 # TODO: 1?
                    })

                cfg_dict_list.append(syst_cfg)

    # write systematic run config to file
    outname_config_syst = f"{outname_config}_syst.json"
    util.write_dict_to_json(cfg_dict_list, outname_config_syst)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--sample-dir", type=str,
                        default="~/atlasserv/NtupleTT/latest",
                        help="Sample directory")
    parser.add_argument("-n", "--config-name", type=str,
                        default="runConfig")
    parser.add_argument("-c", "--category", choices=["ejets", "mjets", "ljets"],
                        default="ljets")
    parser.add_argument("-r", "--result-dir", type=str,
                        default="~/data/OmniFoldOutputs/Run2",
                        help="Output directory of unfolding runs")
    parser.add_argument("-e", "--subcampaigns", nargs='+', choices=["mc16a", "mc16d", "mc16e"], default=["mc16a", "mc16d", "mc16e"])
    parser.add_argument("-s", "--systematics", type=str, nargs="*", default=[],
                        help="List of systematic uncertainties to evaluate. A special case: 'all' includes all systematics")
    parser.add_argument("-b", "--do-bootstrap", action="store_true",
                        help="If True, also generate run configs to do bootstrap")

    args = parser.parse_args()

    # hard code common config here for now
    common_cfg = {
        "observable_config" : "configs/observables/vars_ttbardiffXs_pseudotop.json",
        "binning_config" : "configs/binning/bins_ttdiffxs.json",
        "iterations" : 4,
        "batch_size" : 20000,
        "normalize" : False,
        "nruns" : 7,
        "parallel_models" : 3,
        "resample_data" : False,
        "correct_acceptance" : True
    }

    createRun2Config(
        args.sample_dir,
        category = args.category,
        outname_config = args.config_name,
        output_top_dir = args.result_dir,
        subcampaigns = args.subcampaigns,
        do_bootstrap = args.do_bootstrap,
        systematics = args.systematics,
        common_cfg = common_cfg
        )
