import os
import h5py
import numpy as np
from numpy.random import default_rng
rng = default_rng()

from datahandler_base import DataHandlerBase, filter_filepaths, filter_variable_names
from hadd_h5 import hadd_h5

import logging
logger = logging.getLogger('datahandler_h5')

weight_components = ["weight_bTagSF_DL1r_70", "weight_jvt", "weight_leptonSF", "weight_pileup", "weight_mc"]

def get_dataset_h5(file_h5, name, prefix=''):
    if name in file_h5:
        return file_h5[name]
    elif f"{prefix}.{name}" in file_h5:
        return file_h5[f"{prefix}.{name}"]
    else:
        raise RuntimeError(f"Cannot find variable {name} in the dataset")

def get_weight_variables(weight_name_nominal, weight_type):
    if weight_type == 'nominal' or weight_type.startswith("external:"):
        return weight_name_nominal, None, None

    else: # not nominal weights
        # Examples of expected 'weight_type':
        # 'weight_pileup_UP' or 'weight_bTagSF_DL1r_70_eigenvars_B_up:5'
        if len(weight_type.split(':')) > 1:
            # The weight variation branch is a vector of float
            weight_var, index_w = weight_type.split(':')
            weight_syst = f"{weight_var.strip()}_{index_w.strip()}"
        else:
            weight_syst = weight_type

        # component of the nominal weights corresponding to the weight variation
        # All components of the nominal weights (hard code here for now)
        weight_comp = None
        for wname in weight_components:
            if weight_syst.startswith(wname):
                weight_comp = wname
                break

        if weight_syst == 'mc_generator_weights':
            weight_comp = 'weight_mc'

        if weight_comp is None: # something's wrong
            raise RuntimeError(f"Unknown base component for event weight {weight_type}")

        return weight_name_nominal, weight_syst, weight_comp

def in_MeV_reco(variable_name):
    return variable_name in ['jet_pt', 'jet_e', 'met_met', 'mwt', 'lep_pt', 'lep_m']

def in_MeV_truth(variable_name):
    isMC = variable_name.startswith("MC_")
    isEnergy = variable_name.endswith("_pt") or variable_name.endswith("_m") or variable_name.endswith("_E") or variable_name.endswith("_Ht") or variable_name.endswith("_pout")
    return isMC and isEnergy

class DataHandlerH5(DataHandlerBase):
    """
    Load data from hdf5 files

    Parameters
    ----------
    filepaths :  str or sequence of str
        List of root file names to load
    variable_names : list of str, optional
        List of reco level variable names to read. If not provided, read all
    variable_names_mc : list of str, optional
        List of truth level variable names to read. If not provided, read all
    weight_type : str, default 'nominal'
        Type of event weights to load for systematic uncertainty variations
    treename_reco : str, default 'reco'
        Name of the reconstruction-level tree
    treename_truth : str, default 'parton'
        Name of the truth-level tree. If empty or None, skip loading the 
        truth-level tree
    """
    def __init__(
        self,
        filepaths,
        variable_names=[],
        variable_names_mc=[],
        outputname = 'VDS.h5',
        treename_reco='reco',
        treename_truth='parton',
        weight_name_nominal='normalized_weight',
        weight_type='nominal',
        match_dR = None, # float
        plot_dir = None, # str
        odd_or_even = None, #str, 'odd', 'even'
        ):

        super().__init__()

        ######
        # load data from h5 files
        filepaths_clean, weight_rescale_factors = filter_filepaths(filepaths)

        # for now
        if weight_rescale_factors and not all(f==1. for f in weight_rescale_factors):
            logger.warning("weight rescaling from the file names are currently not supported with hdf5 inputs")

        ###
        if not os.path.splitext(outputname)[-1]: # no extension specified
            outputname = outputname.strip('_')+'_vds.h5'

        # concatenate multiple h5 files into a virtual dataset
        logger.debug(f"Create virtual dataset file: {outputname}")
        self.vds = h5py.File(outputname, 'w')

        self.data_reco = None
        self.data_truth = None

        self._load_arrays(
            filepaths_clean,
            variable_names,
            variable_names_mc,
            prefix_reco = treename_reco,
            prefix_truth = treename_truth
        )

        ######
        # event weights
        logger.debug("Load weight arrays")
        self.weights = None
        self.weights_mc = None

        # reweighter
        self._reweighter = None
        # scale factors
        self._w_sf = 1.

        if weight_type.startswith("external:"):
            # special case: load event weights from external files
            # "external:" is followed by a comma separated list of file paths
            filepaths_w = weight_type.replace("external:","",1).split(",")
        else:
            # load event weights from the same files as data arrays
            filepaths_w = filepaths_clean

        self._load_weights(
            filepaths_w,
            weight_name_nominal = weight_name_nominal,
            weight_type = weight_type,
            include_truth = len(variable_names_mc) > 0
        )

        ######
        # event selection flags
        self.pass_reco = None
        self.pass_truth = None
        self.event_filter = None

        self._set_event_selections(
            filepaths,
            has_truth = len(variable_names_mc) > 0,
            match_dR = match_dR,
            prefix = treename_truth
        )

        ###
        # check length
        assert len(self) == len(self.weights)
        assert len(self) == len(self.pass_reco)

        if self.data_truth:
            assert len(self) == len(self.weights_mc)
            assert len(self) == len(self.pass_truth)

        ####
        # filters
        self.event_filter = self._event_number_filter(odd_or_even)

        if self.event_filter is not None:
            # apply filters only to event selection flags for now
            self.pass_reco = self.pass_reco[self.event_filter]
            if self.data_truth:
                self.pass_truth = self.pass_truth[self.event_filter]

    def __del__(self):
        self.vds.close()

    def __len__(self):
        # assume all variable datasets are of the same size
        vname = list(self.vds.keys())[0]
        return len(self.vds[vname])

    def _get_reco_arr(self, feature, outarr=None):
        if outarr is None:
            outarr = np.zeros(len(self), dtype=np.float32)

        outarr[:] = self.data_reco[feature][:]

        # convert MeV to GeV if needed
        if in_MeV_reco(feature):
            outarr[:] /= 1000.

        return outarr

    def _get_truth_arr(self, feature, outarr=None):
        if outarr is None:
            outarr = np.zeros(len(self), dtype=np.float32)

        outarr[:] = self.data_truth[feature][:]

        # convert MeV to GeV if needed
        if in_MeV_truth(feature):
            outarr /= 1000.

        return outarr

    def _get_reco_keys(self):
        return list(self.data_reco.keys())

    def _get_truth_keys(self):
        return list(self.data_truth.keys())

    def _load_arrays(
        self,
        filepaths,
        variables_reco,
        variables_truth=[],
        prefix_reco='',
        prefix_truth=''
        ):

        # concatenate multiple h5 files into a virtual dataset
        logger.debug(f"Concatenate data arrays from hdf5 files into a virtual dataset")

        # variables
        variables_reco = filter_variable_names(variables_reco)
        if variables_truth:
            variables_truth = filter_variable_names(variables_truth)

        hadd_h5(
            filepaths,
            variable_names = variables_reco + variables_truth,
            outputfile = self.vds,
            prefix = prefix_truth
        )

        self.data_reco = {}
        logger.debug("Load reco-level data array")
        for vname in variables_reco:
            self.data_reco[vname] = get_dataset_h5(self.vds, vname, prefix_reco)

        if variables_truth:
            self.data_truth = {}
            logger.debug("Load truth-level data array")
            for vname in variables_truth:
                self.data_truth[vname] = get_dataset_h5(self.vds, vname, prefix_truth)

    def _load_weights(
        self,
        filepaths,
        weight_name_nominal,
        weight_type = 'nominal',
        include_truth = False
        ):

        if weight_type == 'nominal' or weight_type.startswith("external:"):
            logger.debug("Concatenate event weight files to a virtual dataset")
            hadd_h5(
                filepaths,
                variable_names = [weight_name_nominal],
                outputfile = self.vds
            )

            self.weights = self.vds[weight_name_nominal]

        else: # not nominal weights
            wname_nom, wname_syst, wname_comp = get_weight_variables(weight_name_nominal, weight_type)
            hadd_h5(
                filepaths,
                variable_names = [wname_nom, wname_syst, wname_comp],
                padding = wname_nom, # in case not all sub samples have the weight variables
                outputfile = self.vds
            )

            # weights_nominal * weights_syst / weights_component
            warr = self.vds[wname_nom][:]
            warr *= self.vds[wname_syst]
            np.divide(warr, self.vds[wname_comp], out=warr, where = self.vds[wname_comp][:]!=0)

            # store the new weights
            self.vds.create_dataset('weights', data=warr)

            self.weights = self.vds['weights']

        if include_truth:
            # for now
            self.weights_mc = self.weights

    def _set_event_selections(
        self,
        filepaths,
        has_truth = False,
        match_dR = None, # float
        prefix = ''
        ):

        # variables relevant for event selections
        variables_sel = ['eventNumber', 'pass_reco']

        if has_truth:
            variables_sel += ['pass_truth']
            if match_dR is not None:
                variables_sel += ['dR_thad', 'dR_tlep', 'dR_lq1', 'dR_lq2', 'dR_lep', 'dR_nu']

        hadd_h5(
            filepaths,
            variable_names = variables_sel,
            outputfile = self.vds,
            prefix = prefix
        )

        self.pass_reco = self.vds['pass_reco'][:] # & (self.weights != 0)
        self.pass_truth = self.vds['pass_truth'][:] if has_truth else None

        if match_dR is not None and self.pass_truth is not None:
            # match the top decay products
            match_dR_top_decays = self.vds["dR_lq1"][:] < match_dR & self.vds["dR_lq2"][:] < match_dR & self.vds["dR_lep"][:] < match_dR & self.vds["dR_nu"][:] < match_dR
            self.pass_truth &= match_dR_top_decays
            # or match the top quarks
            #match_dR_tops = self.vds["dR_thad"] < match_dR & self.vds["dR_tlep"] < match_dR
            #self.pass_truth &= match_dR_tops

    def _event_number_filter(self, odd_or_even):
        if odd_or_even == 'odd':
            return self.vds["eventNumber"][:]%2 == 1
        elif odd_or_even == 'even':
            return self.vds["eventNumber"][:]%2 == 0
        elif odd_or_even is not None:
            logger.warning(f"Unknown value for the argument 'odd_or_even': {odd_or_even}. No selection is applied.")
            return None
        else:
            return None

    def _get_array(self, feature, outarr=None):
        if outarr is None:
            outarr = np.zeros(len(self), dtype=np.float32)

        if self._in_data_reco(feature): # reco level
            self._get_reco_arr(feature, outarr)

        elif self._in_data_truth(feature): # truth level
            self._get_truth_arr(feature, outarr)

        # special cases
        elif feature.endswith('px'):
            feature_pt = feature.replace('_px', '_pt')
            feature_phi = feature.replace('_px', '_phi')
            # px = pt * cos(phi)
            outarr[:] = self._get_reco_arr(feature_pt) * np.cos(self._get_reco_arr(feature_phi))
        elif feature.endswith('py'):
            feature_pt = feature.replace('_py', '_pt')
            feature_phi = feature.replace('_py', '_phi')
            # py = pt * sin(phi)
            outarr[:] = self._get_reco_arr(feature_pt) * np.sin(self._get_reco_arr(feature_phi))
        elif feature.endswith('pz'):
            feature_pt = feature.replace('_pz', '_pt')
            feature_eta = feature.replace('_pz', '_eta')
            # pz = pt * sinh(eta)
            outarr[:] = self._get_reco_arr(feature_pt) * np.sinh(self._get_reco_arr(feature_eta))
        else:
            raise RuntimeError(f"Unknown feature {feature}")

        return outarr

    def get_arrays(self, features, valid_only=False):
        arr_shape = (len(self),) + np.asarray(features).shape
        feature_arr = np.zeros(shape=arr_shape, dtype=np.float32)

        if feature_arr.ndim == 1:
            self._get_array(str(features), feature_arr)

        else:
            for i, varname in enumerate(features):
                self._get_array(varname, feature_arr[:,i])

        # event filter
        if self.event_filter is not None:
            feature_arr = feature_arr[self.event_filter]
            # Event selection flags self.pass_reco and self.pass_truth should already have the filter applied in self.__init__()

        if valid_only:
            if self._in_data_reco(features):
                feature_arr = feature_arr[self.pass_reco]
            elif self._in_data_truth(features):
                feature_arr = feature_arr[self.pass_truth]
            else:
                # a mixture
                feature_arr = feature_arr[self.pass_reco & self.pass_truth]

        return feature_arr

    def get_weights(self, bootstrap=False, reco_level=True, valid_only=True):
        w_dataset = self.weights if reco_level else self.weights_mc

        # event filter
        if self.event_filter is not None:
            w_arr = w_dataset[self.event_filter]
        else:
            w_arr = w_dataset[:]

        if self._reweighter is not None:
            # reweight events that pass both reco and truth level selections
            rwtsel = self.pass_reco & self.pass_truth
            v_arr = self.get_arrays(self._reweighter.variables, valid_only=False)[rwtsel]
            w_arr[rwtsel] *= self._reweighter.func(v_arr)

        if valid_only:
            evtsel = self.pass_reco if reco_level else self.pass_truth
            w_arr = w_arr[evtsel]

        # rescale
        w_arr *= self._w_sf

        if bootstrap:
            w_arr *= rng.poisson(1, size=len(w_arr))

        return w_arr

    def sum_weights(self, reco_level=True):
        return self.get_weights(reco_level=reco_level, valid_only=True).sum()

    def rescale_weights(self, factors=1., reweighter=None):
        # for reweighting sample
        if reweighter is not None:
            self._reweighter = reweighter

        # rescale
        self._w_sf *= factors

    def _filter_events_fail_selections(self, event_sel):
        if self.event_filter is None:
            self.event_filter = event_sel.copy()
        else:
            # event selection flags self.pass_reco and self.pass_truth should already be filtered
            # only update a subset of self.event_filter that corresponds to the selection flags
            self.event_filter[self.event_filter] &= event_sel

        # update event selection falgs
        self.pass_reco = self.pass_reco[event_sel]
        if self.pass_truth is not None:
            self.pass_truth = self.pass_truth[event_sel]

    def remove_unmatched_events(self):
        # set filter to remove events that do not pass all selections
        if self.data_truth:
            self._filter_events_fail_selections(self.pass_reco & self.pass_truth)
        else:
            self._filter_events_fail_selections(self.pass_reco)

    def remove_events_failing_reco(self):
        self._filter_events_fail_selections(self.pass_reco)

    def remove_events_failing_truth(self):
        if self.data_truth is None:
            return

        self._filter_events_fail_selections(self.pass_truth)

    def clear_underflow_overflow_events(self):
        notflow = not self.is_underflow_or_overflow()
        self._filter_events_fail_selections(notflow)
        self.reset_underflow_overflow_flags()

    # Use DataHandlerBase
    #def reset_underflow_overflow_flags(self):
    #def update_underflow_overflow_flags(self, varnames, bins):
    #def is_underflow_or_overflow(self):