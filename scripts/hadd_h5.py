import os
import h5py

import logging
logger = logging.getLogger('hadd_h5')

def hadd_h5(
    inputfiles,
    variable_names = [],
    outputname = None,
    outputfile = None,
    padding = None,
    verbose = False,
    prefix = ''
    ):
    """
    Concatenate multiple hdf5 files into one virtual dataset
    """
    # configure the logger
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    nevents = []

    logger.debug("Loop over the input files to get the total number of events")
    for infile in inputfiles:
        with h5py.File(infile, "r") as fin:
            # get all variables from the input file if it is not specified
            if not variable_names:
                variable_names = list(fin.keys())

            # assume the shape of all variables are the same
            nevents.append(len(fin[variable_names[0]]))
    logger.debug(f"Number of events in each input file: {nevents}")

    logger.debug("Loop over the input files again to build the virtual dataset")
    layout_d = {vname : None for vname in variable_names}
    nevts_tot = 0
    for ievt, infile in zip(nevents, inputfiles):
        logger.info(ievt, infile)
        with h5py.File(infile, 'r') as fin:
            for vname in variable_names:
                # check if vname is an available dataset in fin
                if vname in fin:
                    vsource = h5py.VirtualSource(fin[vname])
                elif prefix and f"{prefix}.{vname}" in fin:
                    vsource = h5py.VirtualSource(fin[f"{prefix}.{vname}"])
                elif padding:
                    # Use the dataset 'padding' instead
                    logger.warning(f"No dataset {vname} in file {infile}. Use the specified dataset {padding} to fill the virtual dataset instead.")

                    if not padding in fin:
                        raise KeyError(f"No padding dataset {padding} either!")

                    vsource = h5py.VirtualSource(fin[padding])

                else:
                    raise KeyError(f"No dataset {vname} in file {infile}")

                if layout_d[vname] is None:
                    layout_d[vname] = h5py.VirtualLayout(shape=(sum(nevents),), dtype=fin[vname].dtype)

                layout_d[vname][nevts_tot:nevts_tot+ievt] = vsource

        nevts_tot += ievt

    # Add virtual dataset to file
    fout_exist = outputfile is not None
    if not fout_exist:
        if not outputname:
            output_prefix = os.path.commonprefix(inputfiles)
            outputname = output_prefix.rstrip('_') + "_VDS.h5"

        logger.info(f"Create output file: {outputname}")
        outputfile = h5py.File(outputname, 'w')

    for vname in layout_d:
        outputfile.create_virtual_dataset(vname, layout_d[vname])

    if not fout_exist:
        # close the file
        outputfile.close()

    return outputname

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', "--inputfiles", nargs='+', type=str,
                        help="List of input file paths")
    parser.add_argument("-n", "--variable-names", nargs='+', type=str,
                        help="List of variable names. If not provided, use all available in the dataest")
    parser.add_argument("-o", "--outputname", type=str,
                        help="Output virtual dataest file name")
    parser.add_argument("-p", "--padding", type=str,
                        help="If provided, use this name for making the virtual dataset in case a variable name is not available in some input files")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="If True, set the logging level to debug otherwise info")
    parser.add_argument("--prefix", type=str, default="", help="Prefix of variable name")

    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)

    hadd_h5(**vars(args))