import os
import h5py
import numpy as np

from datahandler_base import DataHandlerBase, filter_filepaths, filter_variable_names
from hadd_h5 import hadd_h5

import logging
logger = logging.getLogger('datahandler_h5')

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
        all_weight_components = ["weight_bTagSF_DL1r_70", "weight_jvt", "weight_leptonSF", "weight_pileup", "weight_mc"]
        for wname in all_weight_components:
            if weight_syst.startswith(wname):
                weight_comp = wname
                break

        if weight_syst == 'mc_generator_weights':
            weight_comp = 'weight_mc'

        if weight_comp is None: # something's wrong
            raise RuntimeError(f"Unknown base component for event weight {weight_type}")
        
        return weight_name_nominal, weight_syst, weight_comp

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
        self.weights_truth = None

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
            weight_type = weight_type
        )

        # for now
        if variable_names_mc:
            self.weights_mc = self.weights.copy()

        ######
        # event selection flags
        self.pass_reco = None
        self.pass_truth = None

        self._set_event_selections(
            filepaths,
            has_truth = len(variable_names_mc) > 0,
            match_dR = match_dR,
            odd_or_even = odd_or_even,
            prefix = treename_truth
        )

        ######
        # sanity check
        assert len(self) == len(self.weights)
        assert len(self) == len(self.pass_reco)

        if self.data_truth is not None:
            assert len(self) == len(self.pass_truth)
            assert len(self) == len(self.weights_mc)

    def __del__(self):
        self.vds.close()

    def __len__(self):
        # assume all variable datasets are of the same size
        vname = list(self.vds.keys())[0]
        return len(self.vds[vname])

    def _get_reco_arr(self, feature):
        return self.data_reco[feature][:]

    def _get_truth_arr(self, feature):
        return self.data_truth[feature][:]

    def _get_reco_keys(self):
        return list(self.data_reco.keys())

    def _get_truth_keys(self):
        return list(self.data_truth.keys())

    def _filter_reco_arr(self, selections):
        for k in self._get_reco_keys():
            self.data_reco[k] = self.data_reco[k][selections]

    def _filter_truth_arr(self, selections):
        for k in self._get_truth_keys():
            self.data_truth[k] = self.data_truth[k][selections]

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

        self.data_truth = {}
        if variables_truth:
            logger.debug("Load truth-level data array")
            for vname in variables_truth:
                self.data_truth[vname] = get_dataset_h5(self.vds, vname, prefix_truth)

    def _load_weights(
        self,
        filepaths,
        weight_name_nominal,
        weight_type = 'nominal',
        ):

        if weight_type == 'nominal' or weight_type.startswith("external:"):
            logger.debug("Concatenate event weight files to a virtual dataset")
            hadd_h5(
                filepaths,
                variable_names = [weight_name_nominal],
                outputfile = self.vds
            )

            self.weights = self.vds[weight_name_nominal][:]

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
            all_weight_components = ["weight_bTagSF_DL1r_70", "weight_jvt", "weight_leptonSF", "weight_pileup", "weight_mc"]
            for wname in all_weight_components:
                if weight_syst.startswith(wname):
                    weight_comp = wname
                    break

            if weight_syst == 'mc_generator_weights':
                weight_comp = 'weight_mc'

            if weight_comp is None: # something's wrong
                raise RuntimeError(f"Unknown base component for event weight {weight_type}")

            logger.debug("Concatenate event weight files to a virtual dataset")
            hadd_h5(
                filepaths,
                variable_names = [weight_name_nominal, weight_syst, weight_comp],
                padding = weight_name_nominal, # in case not all sub samples have the weight variables
                outputfile = self.vds
            )

            # weights_nominal * weights_syst / weights_component
            self.weights = self.vds[weight_name_nominal][:]

            warr_syst = self.vds[weight_syst][:]
            warr_comp = self.vds[weight_comp][:]

            sf = np.zeros_like(self.weights, float)
            np.divide(warr_syst, warr_comp, out = sf, where = warr_comp!=0)

            self.weights *= sf

    def _set_event_selections(
        self,
        filepaths,
        has_truth = False,
        match_dR = None, # float
        odd_or_even = None, # str, 'odd', 'even'
        prefix = ''
        ):

        # variables for selections
        variables_sel = ['pass_reco']

        if has_truth:
            variables_sel += ['pass_truth']
            if match_dR is not None:
                variables_sel += ['dR_thad', 'dR_tlep', 'dR_lq1', 'dR_lq2', 'dR_lep', 'dR_nu']

        if odd_or_even is not None:
            variables_sel += ['eventNumber']

        hadd_h5(
            filepaths,
            variable_names = variables_sel,
            outputfile = self.vds,
            prefix = prefix
        )

        self.pass_reco = self.vds['pass_reco'] & (self.weights != 0)

        self.pass_truth = self.vds['pass_truth'][:] if has_truth else None

        if match_dR is not None and self.pass_truth is not None:
            # match the top decay products
            match_dR_top_decays = self.vds["dR_lq1"][:] < match_dR & self.vds["dR_lq2"][:] < match_dR & self.vds["dR_lep"][:] < match_dR & self.vds["dR_nu"][:] < match_dR
            self.pass_truth &= match_dR_top_decays
            # or match the top quarks
            #match_dR_tops = self.vds["dR_thad"] < match_dR & self.vds["dR_tlep"] < match_dR
            #self.pass_truth &= match_dR_tops

        ###
        # select events based on the event numbers
        if odd_or_even == 'odd':
            sel_evt = self.vds["eventNumber"][:]%2 == 1
        elif odd_or_even == 'even':
            sel_evt = self.vds["eventNumber"][:]%2 == 0
        elif odd_or_even is not None:
            logger.warn(f"Unknown value for the argument 'odd_or_even': {odd_or_even}. No selection is applied.")
            sel_evt = None
        else:
            sel_evt = None

        if sel_evt is not None:
            self.pass_reco &= sel_evt
            if self.pass_truth is not None:
                self.pass_truth &= sel_evt