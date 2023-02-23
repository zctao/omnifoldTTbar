#!/usr/bin/env python3
import os

import histogramming as myhu

# systematic uncertainties
from ttbarDiffXsRun2.systematics import get_systematics

import logging
logger = logging.getLogger('EvaluateUncertainties')

def get_unfolded_histogram_from_dict(
    observable, # str, observable name
    histograms_dict, # dict, histograms dictionary
    nensembles = None, # int, number of ensembles for stablizing unfolding
    ibu = False, # bool, if True, read IBU unfolded distribution
    ):

    if ibu:
        return histograms_dict[observable].get('ibu')

    if nensembles is None:
        # return the one from dict computed from all available ensembles
        return histograms_dict[observable].get('unfolded')
    else:
        hists_allruns = histograms_dict[observable].get('unfolded_allruns')
        if nensembles > len(hists_allruns):
            logger.warn(f"The required number of ensembles {nensembles} is larger than what is available: {len(hists_allruns)}")
            nensembles = len(hists_allruns)

        # Use a subset of the histograms from all runs
        h_unfolded = myhu.average_histograms(hists_allruns[:nensembles])
        h_unfolded.axes[0].label = histograms_dict[observable]['unfolded'].axes[0].label

        return h_unfolded

def compute_model_uncertainties(
    histograms_nominal_d, # dict, nominal unfolded histograms of all observables
    nensembles_model = None, # int, number of runs to compute bin uncertainties. If None, use all available
    ):

    logger.debug("Compute model bin uncertainties")

    model_unc_d = dict()

    for ob in histograms_nominal_d:
        logger.debug(ob)
        model_unc_d[ob] = dict()

        h_unfolded = get_unfolded_histogram_from_dict(ob, histograms_nominal_d, nensembles_model)

        # relative bin errors
        vals, sigmas = myhu.get_values_and_errors(h_unfolded)

        # store it as a histogram
        model_unc_d[ob]['network'] = h_unfolded.copy()
        myhu.set_hist_contents(model_unc_d[ob]['network'], sigmas / vals)
        model_unc_d[ob]['network'].view()['variance'] = 0.
        # set name?

    return model_unc_d

def compute_bootstrap_uncertainties(
    bootstrap_topdir, # str, top directory of the results for bootstraping
    histograms_nominal_d, # dict, nominal unfolded histograms of all observables
    nensembles_model = None, # int, number of runs to compute bin uncertainties. If None, use all available
    hist_filename = "histograms.root",
    nresamples = None # int
    ):

    logger.debug("Compute statistical bin uncertainties from bootstraping")

    # file paths to bootstraping histograms
    fpaths_hists_bootstrap = [os.path.join(bootstrap_topdir, d, hist_filename) for d in os.listdir(bootstrap_topdir)]

    if nresamples is not None: #
        if nresamples > len(fpaths_hists_bootstrap):
            logger.warn(f"Required {nresamples} resamples but only {len(fpaths_hists_bootstrap)} available")
            nresamples = len(fpaths_hists_bootstrap)

        fpaths_hists_bootstrap = fpaths_hists_bootstrap[:nresamples]

    logger.debug("Collect unfolded distributions from bootstraping")
    hists_resample_d = dict() # key: observable; value: a list of histograms

    for fpath in fpaths_hists_bootstrap:
        # read histograms into dict
        hists_d = myhu.read_histograms_dict_from_file(fpath)

        # loop over all observables
        for ob in hists_d:
            # get the unfolded distribution from every resample
            if not ob in hists_resample_d:
                hists_resample_d[ob] = list()

            hists_resample_d[ob].append(
                get_unfolded_histogram_from_dict(ob, hists_d, nensembles_model)
                )

    # compute bin uncertainties
    stat_unc_d = dict()

    for ob, hists_rs in hists_resample_d.items():
        logger.debug(ob)
        stat_unc_d[ob] = dict()

        sigmas_stat = myhu.get_sigma_from_hists(hists_rs)

        # get the nominal bin entries
        h_norminal = get_unfolded_histogram_from_dict(ob, histograms_nominal_d, nensembles_model)

        # relative errors
        relerr_stat = sigmas_stat / h_norminal.values()

        # store as a histogram
        stat_unc_d[ob]['bootstrap'] = h_norminal.copy()
        myhu.set_hist_contents(stat_unc_d[ob]['bootstrap'], relerr_stat)
        stat_unc_d[ob]['bootstrap'].view()['variance'] = 0.
        # set name?

    return stat_unc_d

def compute_systematic_uncertainties(
    systematics_topdir,
    histograms_nominal_d,
    nensembles_model = None,
    systematics_keywords = [],
    hist_filename = "histograms.root",
    every_run = False,
    ibu = False,
    ):

    logger.debug("Compute systematic bin uncertainties")
    syst_unc_d = dict()

    logger.debug("Loop over systematic uncertainty variations")
    for syst_variation in get_systematics(systematics_keywords):
        logger.debug(syst_variation)

        fpath_hist_syst = os.path.join(systematics_topdir, syst_variation, hist_filename)

        # read histograms
        hists_syst_d = myhu.read_histograms_dict_from_file(fpath_hist_syst)

        # loop over observables
        for ob in hists_syst_d:
            logger.debug(ob)
            if not ob in syst_unc_d:
                syst_unc_d[ob] = dict()

            # get the unfolded distributions
            h_syst = get_unfolded_histogram_from_dict(ob, hists_syst_d, ibu=ibu)
            h_nominal = get_unfolded_histogram_from_dict(ob, histograms_nominal_d, nensembles_model, ibu=ibu)

            # compute relative bin errors
            relerr_syst = h_syst.values() / h_nominal.values() - 1.

            # store as a histogram
            syst_unc_d[ob][syst_variation] = h_syst.copy()
            myhu.set_hist_contents(syst_unc_d[ob][syst_variation], relerr_syst)
            syst_unc_d[ob][syst_variation].view()['variance'] = 0.
            # set name?

            if every_run: # compute the systematic variation for every run
                syst_unc_d[ob][f"{syst_variation}_allruns"] = list()

                h_syst_allruns = hists_syst_d[ob].get('unfolded_allruns')
                h_nominal_allruns = histograms_nominal_d[ob].get('unfolded_allruns')

                for h_syst_i, h_nominal_i in zip(h_syst_allruns, h_nominal_allruns):
                    relerr_syst_i = h_syst_i.values() / h_nominal_i.values() - 1.

                    # store as a histogram
                    herrtmp_i = h_syst_i.copy()
                    myhu.set_hist_contents(herrtmp_i, relerr_syst_i)
                    herrtmp_i.view()['variance'] = 0.

                    syst_unc_d[ob][f"{syst_variation}_allruns"].append(herrtmp_i)

    return syst_unc_d

def evaluate_uncertainties(
    nominal_dir, # str, directory of the nominal unfolding results
    bootstrap_topdir = None, # str, top directory of the results for bootstraping
    systematics_topdir = None, # str, top directory of the results for systemaic uncertainties
    output_name = 'bin_uncertainties.root', # str, output file name
    nensembles_model = None, # int, number of runs to compute bin uncertainties. If None, use all available
    nresamples_bootstrap = None, # int, number of resamples for bootstrap. If None, use all available
    systematics_keywords = [], # list of str, keywords for selecting a subset of systematic uncertainties
    systematics_everyrun = False, # boolen
    hist_filename = "histograms.root", # str, name of the histogram root file
    ibu = False
    ):

    bin_uncertainties_d = dict()

    # Read nominal results
    fpath_histograms_nominal = os.path.join(nominal_dir, hist_filename)
    logger.info(f"Read nominal histograms from {fpath_histograms_nominal}")
    hists_nominal_d = myhu.read_histograms_dict_from_file(fpath_histograms_nominal)

    # model uncertainty
    bin_errors_model_d = compute_model_uncertainties(hists_nominal_d, nensembles_model)
    bin_uncertainties_d.update(bin_errors_model_d)

    # statistical uncertainty from bootstraping
    if bootstrap_topdir:
        logger.info(f"Read bootstrap results from {bootstrap_topdir}")

        bin_errors_stat_d = compute_bootstrap_uncertainties(
            bootstrap_topdir,
            hists_nominal_d,
            nensembles_model = nensembles_model,
            nresamples = nresamples_bootstrap,
            hist_filename = hist_filename
            )

        for ob in bin_uncertainties_d:
            bin_uncertainties_d[ob].update(bin_errors_stat_d[ob])

    # systematic uncertainties
    if systematics_topdir:
        logger.info(f"Read results for systematic uncertainty variations from {systematics_topdir}")

        bin_errors_syst_d = compute_systematic_uncertainties(
            systematics_topdir,
            hists_nominal_d,
            nensembles_model = nensembles_model,
            systematics_keywords = systematics_keywords,
            hist_filename = hist_filename,
            every_run = systematics_everyrun,
            ibu = ibu
            )

        for ob in bin_uncertainties_d:
            bin_uncertainties_d[ob].update(bin_errors_syst_d[ob])

    # save to file
    logger.info(f"Write to output file {output_name}")
    myhu.write_histograms_dict_to_file(bin_uncertainties_d, output_name)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("nominal_dir", type=str,
                        help="Directory of the nominal unfolding results")
    parser.add_argument("-b", "--bootstrap-topdir", type=str,
                        help="Top directory of the unfolding results for bootstraping")
    parser.add_argument("-s", "--systematics-topdir", type=str,
                        help="Top directory of the unfolding results for systematic uncertainty variations")
    parser.add_argument("-o", "--output-name", type=str, default="bin_uncertainties.root",
                        help="Output file name")
    parser.add_argument("-n", "--nensembles-model", type=int,
                        help="Number of runs for evaluating model uncertainty. If None, use all available runs")
    parser.add_argument("-r", "--nresamples-bootstrap", type=int,
                        help="Number of resamples for boostrap. If None, use all available")
    parser.add_argument("-k", "--systematics-keywords", nargs='*', type=str,
                        help="Keywords for selecting a subset of systematic uncertainties")
    parser.add_argument("--systematics-everyrun", action='store_true',
                        help="If True, compute the systematic bin uncertainties for every unfolding run")
    parser.add_argument("--hist-filename", type=str, default="histograms.root",
                        help="Name of the unfolding histogram file")
    parser.add_argument("--ibu", action='store_true',
                        help="If True, use unfolded distributions from IBU for debugging")

    args = parser.parse_args()

    evaluate_uncertainties(**vars(args))