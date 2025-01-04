import os

from datahandler_toy import DataHandlerToy
from datahandler_root import DataHandlerROOT
from datahandler_h5 import DataHandlerH5

class DataHandlerFactory:
    def __init__(self):
        self._handlers = {}

    def register(self, key, handler):
        self._handlers[key] = handler

    def get(self, key, *args, **kwargs):
        handler = self._handlers.get(key)
        if not handler:
            raise ValueError(f"Unknown data handler: {key}")
        return handler(*args, **kwargs)

dhFactory = DataHandlerFactory()
dhFactory.register('toy', DataHandlerToy)
dhFactory.register('root', DataHandlerROOT)
dhFactory.register('hdf5', DataHandlerH5)

def getDataHandler(
    filepaths, # list of str
    variables_reco, # list of str
    variables_truth = [], # list of str
    reweighter = None,
    use_toydata = False,
    outputname = '.',
    **kwargs
    ):
    """
    Get and load a datahandler according to the input file type
    """

    if use_toydata:
        dh = dhFactory.get('toy')
        dh.load_data(filepaths)

    elif ".root" in filepaths[0]:
        # ROOT files
        dh = dhFactory.get("root", filepaths, variables_reco, variables_truth, **kwargs)

    elif ".h5" in filepaths[0]:
        # HDF5 files
        dh = dhFactory.get("hdf5", filepaths, variables_reco, variables_truth, outputname=outputname, **kwargs)

    else:
        #raise ValueError(f"No data handler for files with extension {input_ext}")
        raise ValueError(f"No data handler for files e.g. {filepaths[0]}")

    if reweighter is not None:
        # TODO: check if variables required by reweighter are included
        dh.rescale_weights(reweighter=reweighter)

    return dh