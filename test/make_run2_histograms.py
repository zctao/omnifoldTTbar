import os
import logging

import util
from make_histograms import make_histograms

import argparse

parser = argparse.ArgumentParser(
    description="Run scripts/make_histograms.py for all unfolding results of Run2 datasets")

parser.add_argument("top_result_dir", type=str,
                    help="Top directory that contains the unfolding results")

parser.add_argument("--binning-config", type=str,
                    default='configs/binning/bins_10equal.json',
                    #default='configs/binning/bins_ttdiffxs.json',
                    help="Path to the binning config file for variables.")
parser.add_argument("-f", "--outfilename", type=str, default="histograms.root",
                    help="Output file name")
parser.add_argument("--overwrite", action="store_true",
                    help="If True, overwrite the existing histogram file with th same name specified by --outfilename ")
parser.add_argument("-k", "--resdir-keywords", nargs='+',
                    default=['output_run2'],
                    help="Keywords to match the result directory names. If multiple keywords are provided, only directories containing all the keywords are selected.")

args = parser.parse_args()

logger = logging.getLogger("make_run2_histograms")
util.configRootLogger()

for cwd, subdirs, files in os.walk(args.top_result_dir):
    if not files:
        continue

    if not 'arguments.json' in files or not 'weights_unfolded.npz' in files:
        # cwd is not a directory containing unfolding results
        continue

    # check if the current working directory name contains the keywards
    matched = True
    for kw in args.resdir_keywords:
        matched &= kw in cwd

    if not matched:
        continue

    logger.info(f"Make histograms for {cwd}")

    # check if the output histogram file is already made
    hasHistFile = args.outfilename in files
    if hasHistFile:
        logger.info(f"  {args.outfilename} already exists")
        if args.overwrite:
            logger.info("  Overwrite the existing one ...")
        else:
            logger.info("  Skip ...")
            continue

    make_histograms(
        cwd,
        binning_config = args.binning_config,
        outfilename = args.outfilename,
        include_ibu = False, #
        verbose = False
        )
