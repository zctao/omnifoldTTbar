import numpy as np

import histUtils as myhu
import make_histograms as mh
from ttbarDiffXsRun2.binnedCorrections import apply_acceptance_correction, apply_efficiency_correction

def unfold(
    response,
    h_obs,
    h_prior,
    niterations,
    acceptance_correction=None,
    efficiency_correction=None
    ):

    # apply acceptance correction if available
    if acceptance_correction is not None:
        h_obs = apply_acceptance_correction(h_obs, acceptance_correction)

    hists_unfold = [h_prior]

    for i in range(niterations):
        m = response.values() * hists_unfold[-1].values()
        m /= (m.sum(axis=1)[:,np.newaxis] + 10**-50)

        i_unfold = np.dot(m.T, h_obs.values())

        h_ibu = h_prior.copy()
        h_ibu.view()['value'] = i_unfold
        h_ibu.view()['variance'] = 0.

        hists_unfold.append(h_ibu)

    # exclude h_prior
    hists_unfold = hists_unfold[1:]

    if efficiency_correction is not None:
        hists_unfold = [apply_efficiency_correction(huf, efficiency_correction) for huf in hists_unfold]

    return hists_unfold # shape: (niterations, )

def run_ibu_from_unfolder(
    unfolder,
    vname_reco,
    vname_truth,
    bins_reco,
    bins_truth,
    niterations = 4, # number of iterations
    nresamples = 25, # number of resamples for estimating uncertainties
    all_iterations = False, # if True, return results at every iteration
    norm = None,
    density = False,
    absoluteValue = False,
    acceptance = None,
    efficiency = None,
    flow = False
    ):

    ###
    # observed distribution
    h_obs = mh.get_observed_distribution(unfolder, vname_reco, bins_reco, absoluteValue=absoluteValue, subtract_background=True)

    ###
    # response
    resp = unfolder.handle_sig.get_response(vname_reco, vname_truth, bins_reco, bins_truth, absoluteValue=absoluteValue)

    # prior distribution
    rd2 = unfolder.handle_sig.get_histogram2d(
        vname_reco, vname_truth, 
        bins_reco, bins_truth, 
        absoluteValue_x=absoluteValue, absoluteValue_y=absoluteValue
        )
    h_prior = myhu.projectToYaxis(rd2, flow=flow)

    # run unfolding
    hists_ibu = unfold(
        resp, h_obs, h_prior, niterations, 
        acceptance_correction=acceptance, 
        efficiency_correction=efficiency
        )

    # bin errors and correlation
    hists_ibu_resample = []

    for rs in range(nresamples):

        h_obs_rs = mh.get_observed_distribution(unfolder, vname_reco, bins_reco, absoluteValue=absoluteValue, bootstrap=True, subtract_background=True)

        hists_ibu_resample.append(
            unfold(resp, h_obs_rs, h_prior, niterations, acceptance_correction=acceptance, efficiency_correction=efficiency)
            )

    # standard deviation of each bin
    bin_errors = myhu.get_sigma_from_hists(hists_ibu_resample) # shape: (niterations, nbins_hist)

    # set error
    myhu.set_hist_errors(hists_ibu, bin_errors)

    # normalization
    if norm is not None:
        for h in hists_ibu:
            h = myhu.renormalize_hist(h, norm, density=False)

    if density:
        for h in hists_ibu:
            h /= myhu.get_hist_widths(h)

    # bin correlations
    bin_corr = myhu.get_bin_correlations_from_hists(hists_ibu_resample)

    # Return results
    if all_iterations:
        return hists_ibu, bin_corr, resp
    else:
        return hists_ibu[-1], bin_corr[-1], resp

def run_ibu_from_unfolder_multidim(
    unfolder,
    vnames_reco,
    vnames_truth,
    bins_reco,
    bins_truth,
    niterations = 4, # number of iterations
    nresamples = 25, # number of resamples for estimating uncertainties
    all_iterations = False, # if True, return results at every iteration
    norm = None,
    density = False,
    absoluteValues = False,
    acceptance = None,
    efficiency = None,
    flow = False
    ):

    if not isinstance(absoluteValues, list):
        absoluteValues = [absoluteValues] * len(vnames_reco)

    acceptance_flat = acceptance.flatten() if acceptance else None
    efficiency_flat = efficiency.flatten() if efficiency else None

    ###
    # observed distribution
    fh_obs = mh.get_observed_distribution_multidim(unfolder, vnames_reco, bins_reco, absoluteValues=absoluteValues, subtract_background=True)

    ###
    # prior distribution and response
    fresp = unfolder.handle_sig.get_response_flattened(
        vnames_reco,
        vnames_truth,
        bins_reco,
        bins_truth,
        absoluteValues = absoluteValues,
        normalize_truthbins = False
    )

    # project to Y axis before normalizing the truth bins
    fh_prior = fresp.projectToTruth(flow=flow)

    # normalize each truth bin
    fresp.normalize_truth_bins()

    # run unfolding
    hists_ibu = unfold(
        fresp.get(),
        fh_obs.flatten(),
        fh_prior.flatten(),
        niterations,
        acceptance_correction = acceptance_flat,
        efficiency_correction = efficiency_flat
    )

    # bin errors and correlation
    hists_ibu_resample = []

    for rs in range(nresamples):

        fh_obs_rs = mh.get_observed_distribution_multidim(unfolder, vnames_reco, bins_reco, absoluteValues=absoluteValues, bootstrap=True, subtract_background=True)

        hists_ibu_resample.append(
            unfold(
                fresp.get(),
                fh_obs_rs.flatten(),
                fh_prior.flatten(),
                niterations,
                acceptance_correction = acceptance_flat,
                efficiency_correction = efficiency_flat
            )
        )

    # standard deviation of each bin
    bin_errors = myhu.get_sigma_from_hists(hists_ibu_resample) # shape: (niterations, nbins_hist)

    # set error
    myhu.set_hist_errors(hists_ibu, bin_errors)

    # repack as FlattenedHistogram
    fhists_ibu = []
    for h_ibu in hists_ibu:
        fhists_ibu.append(fh_prior.copy())

        fhists_ibu[-1].fromFlat(h_ibu)

        if norm is not None:
            fhists_ibu[-1].renormalize(norm=norm, density=False)

        if density:
            fhists_ibu[-1].make_density()

    # bin correlations
    bin_corr = myhu.get_bin_correlations_from_hists(hists_ibu_resample)

    # Return results
    if all_iterations:
        return fhists_ibu, bin_corr, fresp
    else:
        return fhists_ibu[-1], bin_corr[-1], fresp