#!/usr/bin/env python3
import os
import glob
import util

# systematics dictionary
from ttbarDiffXsRun2.systematics import get_systematics
#from ttbarDiffXsRun2.systematics import syst_dict, select_systematics

def subCampaigns_to_years(subcampaigns):
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

    return years

def get_samples_data(
    sample_dir, # top direcotry to look for sample files
    category = 'ljets', # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    check_exist = True, # If True, check if the files exist
    ):

    years = subCampaigns_to_years(subcampaigns)

    if isinstance(category, str):
        category = [category]

    data = [os.path.join(sample_dir, f"obs/{y}/data_0_pseudotop_{c}.root") for c in category for y in years]

    assert(data)
    if check_exist:
        for d in data:
            assert(os.path.isfile(d))

    return data

def get_samples_signal(
    sample_dir, # top direcotry to look for sample files
    category = 'ljets', # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    sample_type = 'systCRL', # systCRL or detNP or mcWAlt
    sample_suffix = 'nominal',
    check_exist = True, # If True, check if the files exist
    ):

    if isinstance(category, str):
        category = [category]

    sample_name = f"{sample_type}/ttbar_{sample_suffix}"

    samples_sig = []
    for e in subcampaigns:
        for c in category:
            s = glob.glob(os.path.join(sample_dir, f"{sample_name}/{e}/ttbar_*_pseudotop_parton_{c}.root"))
            s.sort()
            samples_sig += s

    assert(samples_sig)
    if check_exist:
        for f in samples_sig:
            assert(os.path.isfile(f))

    return samples_sig

def get_samples_backgrounds(
    sample_dir, # top direcotry to look for sample files
    category = 'ljets', # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    backgrounds = ['fakes', 'Wjets', 'Zjets', 'singleTop', 'ttH', 'ttV', 'VV'],
    sample_type = 'systCRL', # systCRL or detNP or mcWAlt
    sample_suffix = 'nominal',
    check_exist = True, # If True, check if the files exist
    ):

    if isinstance(category, str):
        category = [category]

    samples_bkg = []

    for bkg in backgrounds:
        if bkg.lower() == "fakes":
            # QCD
            years = subCampaigns_to_years(subcampaigns)
            samples_bkg += [os.path.join(sample_dir, f"fakes/{y}/data_0_pseudotop_{c}.root") for c in category for y in years]
        else:
            if bkg in ["Wjets", "Zjets"]:
                sample_name = f"systCRL/{bkg}_nominal" # only one that is available
            else:
                sample_name = f"{sample_type}/{bkg}_{sample_suffix}"

            samples_bkg += [os.path.join(sample_dir, f"{sample_name}/{e}/{bkg}_0_pseudotop_{c}.root") for c in category for e in subcampaigns]

    assert(samples_bkg)
    if check_exist:
        for f in samples_bkg:
            assert(os.path.isfile(f))

    return samples_bkg

def write_config_nominal(
    sample_local_dir,
    category = "ljets", # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {}
    ):
    print("nominal")

    # list of samples
    data_nominal = get_samples_data(sample_local_dir, category, subcampaigns)
    sig_nominal = get_samples_signal(sample_local_dir, category, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, category, subcampaigns)

    # output directory
    outdir_nominal = os.path.join(output_top_dir, "nominal")

    # config
    nominal_cfg = common_cfg.copy()
    nominal_cfg.update({
        "data": data_nominal,
        "signal": sig_nominal,
        "background": bkg_nominal,
        "outputdir": outdir_nominal,
        "plot_verbosity": 2
        })

    # write run configuration to file
    outname_config_nominal = f"{outname_config}_nominal.json"
    util.write_dict_to_json(nominal_cfg, outname_config_nominal)

def write_config_bootstrap(
    sample_local_dir,
    nresamples,
    start_index = 0,
    category = "ljets", # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {}
    ):
    print("bootstrap data")

    # list of samples
    data_nominal = get_samples_data(sample_local_dir, category, subcampaigns)
    sig_nominal = get_samples_signal(sample_local_dir, category, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, category, subcampaigns)

    # output directory
    outdir_bs = os.path.join(output_top_dir, "bootstrap")
    outdir_bs_dict = {
        f"resamples{n}": outdir_bs for n in range(start_index, start_index+nresamples)
    }

    # config
    bootstrap_cfg = common_cfg.copy()
    bootstrap_cfg.update({
        "data": data_nominal,
        "signal": sig_nominal,
        "background": bkg_nominal,
        "outputdir": outdir_bs_dict,
        "resample_data": True,
        "run_ibu": False
        })

    # write run configuration to file
    outname_config_bootstrap = f"{outname_config}_bootstrap.json"
    util.write_dict_to_json(bootstrap_cfg, outname_config_bootstrap)

def write_config_bootstrap_mc(
    sample_local_dir,
    nresamples,
    start_index = 0,
    category = "ljets", # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {}
    ):
    print("bootstrap mc")

    # list of samples
    data_nominal = get_samples_data(sample_local_dir, category, subcampaigns)
    sig_nominal = get_samples_signal(sample_local_dir, category, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, category, subcampaigns)

    # output directory
    outdir_bs = os.path.join(output_top_dir, "bootstrap_mc")
    outdir_bs_dict = {
        f"resamples{n}": outdir_bs for n in range(start_index, start_index+nresamples)
    }

    # config
    bootstrap_mc_cfg = common_cfg.copy()
    bootstrap_mc_cfg.update({
        "data": data_nominal,
        "signal": sig_nominal,
        "background": bkg_nominal,
        "outputdir": outdir_bs_dict,
        "resample_data": False,
        "resample_mc": True,
        "run_ibu": False
        })

    # write run configuration to file
    outname_config_bootstrap_mc = f"{outname_config}_bootstrap_mc.json"
    util.write_dict_to_json(bootstrap_mc_cfg, outname_config_bootstrap_mc)

def write_config_bootstrap_mc_clos(
    sample_local_dir,
    nresamples,
    start_index = 0,
    category = "ljets", # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {}
    ):
    print("bootstrap mc closure")

    # nominal samples:
    sig_nominal = get_samples_signal(sample_local_dir, category, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, category, subcampaigns)

    # output directory
    outdir_bs = os.path.join(output_top_dir, "bootstrap_mc_clos")
    outdir_bs_dict = {
        f"resamples{n}": outdir_bs for n in range(start_index, start_index+nresamples)
    }

    # config
    bootstrap_mc_cfg = common_cfg.copy()
    bootstrap_mc_cfg.update({
        "data": sig_nominal,
        "bdata": bkg_nominal,
        "signal": sig_nominal,
        "background": bkg_nominal,
        "outputdir": outdir_bs_dict,
        "resample_data": True,
        "resample_mc": False,
        "run_ibu": False,
        "correct_acceptance" : False,
        #"resample_everyrun" : True
        })

    # write run configuration to file
    outname_config_bootstrap_mc = f"{outname_config}_bootstrap_mc_clos.json"
    util.write_dict_to_json(bootstrap_mc_cfg, outname_config_bootstrap_mc)

def write_config_mc_clos(
    sample_local_dir,
    category = "ljets", # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {}
    ):
    print("mc closure")

    # nominal samples:
    sig_nominal = get_samples_signal(sample_local_dir, category, subcampaigns)
    bkg_nominal = get_samples_backgrounds(sample_local_dir, category, subcampaigns)

    # output directory
    outdir_clos = os.path.join(output_top_dir, "mc_clos")

    # config
    mc_clos_cfg = common_cfg.copy()
    mc_clos_cfg.update({
        "data": sig_nominal,
        "bdata": bkg_nominal,
        "signal": sig_nominal,
        "background": bkg_nominal,
        "outputdir": outdir_clos,
        "resample_data": True,
        "resample_mc": False,
        "run_ibu": False,
        "correct_acceptance" : False,
        #"resample_everyrun" : True
        })

    # write run configuration to file
    outname_config_mc_clos = f"{outname_config}_mc_clos.json"
    util.write_dict_to_json(mc_clos_cfg, outname_config_mc_clos)

def write_config_systematics(
    sample_local_dir,
    systematics_keywords = [],
    category = "ljets", # "ejets" or "mjets" or "ljets"
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {}
    ):

    cfg_dict_list = []

    # nominal samples:
    sig_nom = get_samples_signal(sample_local_dir, category, subcampaigns)
    bkg_nom = get_samples_backgrounds(sample_local_dir, category, subcampaigns)

    # systematics as alternative sets of events in TTrees
    for syst in get_systematics(systematics_keywords, syst_type="Branch"):
        print(syst)

        # samples
        # varied samples as pseudo data
        sig_syst = get_samples_signal(
            sample_local_dir, category, subcampaigns,
            sample_type = 'detNP',
            sample_suffix = syst
            )

        # background samples to be mixed with the above signal samples to make pseudo data
        bkg_syst = get_samples_backgrounds(
            sample_local_dir, category, subcampaigns,
            sample_type = 'detNP',
            sample_suffix = syst
            )

        # unfold using the nominal samples

        # output directory
        outdir_syst = os.path.join(output_top_dir, syst)

        syst_cfg = common_cfg.copy()
        syst_cfg.update({
            "data": sig_syst,
            "bdata": bkg_syst,
            "signal": sig_nom,
            "background": bkg_nom,
            "outputdir": outdir_syst,
            "correct_acceptance" : False,
            #"load_models": ?
            #"nruns": ?
            })

        cfg_dict_list.append(syst_cfg)

    # systematics as scale factor variations
    for syst, wtype in zip(*get_systematics(systematics_keywords, syst_type="ScaleFactor", get_weight_types=True)):
        print(syst)

        # output directory
        outdir_syst = os.path.join(output_top_dir, syst)

        syst_cfg = common_cfg.copy()

        # use nominal samples but different weights as pseudo data
        syst_cfg.update({
            "data": sig_nom,
            "bdata": bkg_nom,
            "signal": sig_nom,
            "background": bkg_nom,
            "weight_data": wtype,
            "weight_mc": "nominal",
            "outputdir": outdir_syst,
            "correct_acceptance" : False,
            #"load_models": ?
            #"nruns": ?
            #"unfolded_weights": ?
            })

        cfg_dict_list.append(syst_cfg)

    # TODO modelling uncertainties

    print(f"Number of run configs for systematics: {len(cfg_dict_list)}")

    # write run configs to file
    outname_config_syst = f"{outname_config}_syst.json"
    util.write_dict_to_json(cfg_dict_list, outname_config_syst)

def write_config_models(
    sample_local_dir,
    category = "ljets",
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {}
    ):

    print("Model tests")

    # alternative ttbar vs nominal ttbar
    # samples
    signal_nominal = get_samples_signal(
        sample_local_dir, category, subcampaigns,
        sample_type = 'mcWAlt',
        sample_suffix = 'AFII_nominal'
    )

    for ttbar_alt in ['hw', 'amc']:

        signal_alt = get_samples_signal(
            sample_local_dir, category, subcampaigns,
            sample_type = 'mcWAlt',
            sample_suffix = f"{ttbar_alt}_nominal"
        )

        outdir_alt = os.path.join(output_top_dir, f"ttbar_{ttbar_alt}_vs_nominal")

        # config
        ttbar_alt_cfg = common_cfg.copy()
        ttbar_alt_cfg.update({
            "data": signal_alt,
            "signal": signal_nominal,
            "outputdir": outdir_alt,
            "plot_verbosity": 2,
            "normalize": True,
            "correct_acceptance": False,
            "truth_known": True
        })

        # write run config to file
        outname_config_alt = f"{outname_config}_ttbar_{ttbar_alt}_vs_nominal.json"
        util.write_dict_to_json(ttbar_alt_cfg, outname_config_alt)

def write_config_stress(
    sample_local_dir,
    fpath_reweights = [],
    category = "ljets",
    subcampaigns = ["mc16a", "mc16d", "mc16e"],
    output_top_dir = '.',
    outname_config =  'runConfig',
    common_cfg = {}
    ):

    print("Stress tests")

    sig_nominal = get_samples_signal(sample_local_dir, category, subcampaigns)

    stress_common_cfg = common_cfg.copy()
    stress_common_cfg.update({
        "data": sig_nominal,
        "signal": sig_nominal,
        "plot_verbosity": 2,
        "normalize": True,
        "correct_acceptance": False,
        "truth_known": True
    })

    ######
    # linear th_pt
    stress_th_pt_cfg = stress_common_cfg.copy()
    stress_th_pt_cfg.update({
        "outputdir": os.path.join(output_top_dir, f"stress_th_pt"),
        "reweight_data": "linear_th_pt"
    })

    # write run config to file
    util.write_dict_to_json(stress_th_pt_cfg, f"{outname_config}_stress_th_pt.json")

    ######
    # mtt bump
    stress_bump_cfg = stress_common_cfg.copy()
    stress_bump_cfg.update({
        "outputdir": os.path.join(output_top_dir, f"stress_bump"),
        "reweight_data": "gaussian_bump"
    })

    # write run config to file
    util.write_dict_to_json(stress_bump_cfg, f"{outname_config}_stress_bump.json")

    # data
    if not fpath_reweights:
        print("WARNING cannot generate run config for data induced stress test: no external weight files are provided.")
    else:
        # reweighted signal MC as pseudo-data
        stress_data_cfg = stress_common_cfg.copy()
        stress_data_cfg.update({
            "outputdir": os.path.join(output_top_dir, f"stress_data"),
            "weight_data": f"external:{','.join(fpath_reweights)}",
            "weight_mc": "nominal"
        })

        # write run config to file
        util.write_dict_to_json(stress_data_cfg, f"{outname_config}_stress_data.json")

        # use the reweighted signal MC to unfold data
        data_nominal = get_samples_data(sample_local_dir, category, subcampaigns)
        stress_data_alt_cfg = stress_common_cfg.copy()
        stress_data_alt_cfg.update({
            "outputdir": os.path.join(output_top_dir, f"stress_data_alt"),
            "data": data_nominal,
            "signal": sig_nominal,
            "plot_verbosity": 2,
            "normalize": False,
            "correct_acceptance": True,
            "truth_known": False,
            "weight_data": "nominal",
            "weight_mc": f"external:{','.join(fpath_reweights)}"
        })

        # write run config to file
        util.write_dict_to_json(stress_data_alt_cfg, f"{outname_config}_stress_data_alt.json")

def createRun2Config(
        sample_local_dir,
        category, # "ejets" or "mjets" or "ljets"
        outname_config = 'runConfig',
        output_top_dir = '.',
        subcampaigns = ["mc16a", "mc16d", "mc16e"],
        run_list = None,
        systematics_keywords = [],
        external_reweights = [],
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

    # common arguments for write_config_*
    write_common_args = {
        'sample_local_dir': sample_local_dir,
        'category': category,
        'subcampaigns': subcampaigns,
        'output_top_dir': output_top_dir,
        'outname_config': outname_config,
        'common_cfg': common_cfg
        }

    # nominal
    if 'nominal' in run_list:
        write_config_nominal(**write_common_args)

    # bootstrap for statistical uncertainties
    if 'bootstrap' in run_list:
        write_config_bootstrap(
            nresamples = 10,
            start_index = 0,
            **write_common_args
            )

        write_config_bootstrap_mc(
            nresamples = 10,
            start_index = 0,
            **write_common_args
            )

    # Systematic uncertainties
    if 'systematics' in run_list:
        write_config_systematics(
            systematics_keywords = systematics_keywords,
            **write_common_args
        )

    # MC closure
    if 'closure' in run_list:
        write_config_mc_clos(**write_common_args)

        if 'bootstrap' in run_list:
            write_config_bootstrap_mc_clos(
                nresamples = 10,
                start_index = 0,
                **write_common_args
                )

    if 'model' in run_list:
        write_config_models(**write_common_args)

    if 'stress' in run_list:
        write_config_stress(
            fpath_reweights = external_reweights,
            **write_common_args
        )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--sample-dir", type=str, action=util.ParseEnvVar,
                        default="${DATA_DIR}/NtupleTT/latest",
                        help="Sample directory")
    parser.add_argument("-n", "--config-name", type=str,
                        default="configs/run/ttbarDiffXsRun2/runCfg_run2_ljets")
    parser.add_argument("-c", "--category", choices=["ejets", "mjets", "ljets"],
                        default="ljets")
    parser.add_argument("-r", "--result-dir", type=str, action=util.ParseEnvVar,
                        default="${DATA_DIR}/OmniFoldOutputs/Run2",
                        help="Output directory of unfolding runs")
    parser.add_argument("-e", "--subcampaigns", nargs='+', choices=["mc16a", "mc16d", "mc16e"], default=["mc16a", "mc16d", "mc16e"])
    parser.add_argument("--observables", nargs='+',
                        default=['th_pt', 'th_y', 'tl_pt', 'tl_y', 'ptt', 'ytt', 'mtt'],
                        help="List of observables to unfold")

    run_options = ['nominal', 'bootstrap', 'systematics', 'model', 'closure', 'stress']
    parser.add_argument("-l", "--run-list", nargs="+",
                        choices=run_options, default=run_options,
                        help="List of run types to generate config files. If None, generate run configs for all types")

    parser.add_argument("-k", "--systematics-keywords", type=str, nargs="*", default=[],
                        help="List of keywords to filter systematic uncertainties to evaluate. If empty, include all available.")

    parser.add_argument("--external-reweights",
                        type=str, nargs='+', default=[], action=util.ParseEnvVar,
                        help="List of path to external weight files from reweighting")

    args = parser.parse_args()

    # hard code common config here for now
    common_cfg = {
        "observable_config" : "${SOURCE_DIR}/configs/observables/vars_ttbardiffXs_pseudotop.json",
        "binning_config" : "${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json",
        "iterations" : 4,
        "batch_size" : 20000,
        "normalize" : False,
        "nruns" : 5,
        "parallel_models" : 4,
        "resample_data" : False,
        "correct_acceptance" : True,
        "run_ibu": True,
    }

    if args.observables:
        common_cfg["observables"] = args.observables

    createRun2Config(
        args.sample_dir,
        category = args.category,
        outname_config = args.config_name,
        output_top_dir = args.result_dir,
        subcampaigns = args.subcampaigns,
        run_list = args.run_list,
        systematics_keywords = args.systematics_keywords,
        external_reweights = args.external_reweights,
        common_cfg = common_cfg
        )