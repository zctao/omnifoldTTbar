import os
import tarfile

import util

from datahandler_base import filter_filepaths

def generate_slurm_jobs(
    config_name, # file name of the run config
    sample_dir, # top direcotry for sample files
    sample_tarballs, # a list of file paths to input dataset tarballs
    email = os.getenv('USER')+'@phas.ubc.ca', # for now
    account = "def-alister",
    output_name = None, # slurm job file name,
    output_dir = None, # output directory
    check_tarfiles = True, # if True, check if all input files are available in tarballs
    ):

    if output_dir is None:
        # creat a 'slurm' directory in the same directory as the config file
        output_dir = os.path.join(os.path.dirname(config_name), 'slurm')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    if not output_name:
        # set it to the same as config_name but with a different extension
        output_name = os.path.splitext(os.path.basename(config_name))[0]

    jobfile_name = os.path.join(output_dir, output_name+'.slurm')
    jobcfg_name = os.path.join(output_dir, output_name+'.json')

    # Load run config
    runcfg = util.read_dict_from_json(config_name, parse_env=False)

    # Modify run config for batch jobs
    inputdir_job = './inputs'
    outputdir_job = './outputs'

    # inputs
    for key in ['data', 'signal', 'background', 'bdata']:
        samples = runcfg.get(key)
        if samples:
            # replace the absolute paths of samples with the paths on the node
            samples_job = [os.path.join(inputdir_job, os.path.relpath(sample, sample_dir)) for sample in samples]
            runcfg[key] = samples_job

    # output
    outdir = runcfg['outputdir']
    if not os.path.isdir(outdir):
        print(f"Create output directory {outdir}")
        os.makedirs(outdir)

    result_tarball = os.path.join(outdir, "results.tar")

    # replace the output directory in the config with a local directory on the node
    runcfg['outputdir'] = outputdir_job

    # write the new config file
    util.write_dict_to_json(runcfg, jobcfg_name)

    # Load slurm job template
    slurm_template = os.path.expandvars("${SOURCE_DIR}/cedar/slurmJob.template")
    with open(slurm_template) as ftmp:
        job_str = ftmp.read()

    # Check the list of tarballs
    # Check if file exists, expand to absolute paths if necessary, remove duplicates
    tarballs_set = set()
    for sample_tar in sample_tarballs:
        assert os.path.isfile(sample_tar), f"Cannot find file {sample_tar}!"
        tarballs_set.add(os.path.abspath(sample_tar))

    if check_tarfiles:
        all_sample_files = []
        for sample_tar in tarballs_set:
            with tarfile.open(sample_tar) as ftar:
                for tarinfo in ftar:
                    all_sample_files.append(os.path.join(inputdir_job, tarinfo.name))

        # check if all files in the config are available
        sample_files_config = []
        for key in ['data', 'signal', 'background', 'bdata']:
            samples = runcfg.get(key)
            if not samples:
                continue

            # clean up the sample names
            samples_clean = filter_filepaths(samples)[0]

            sample_files_config += samples_clean

        # sample_files_config should be a subset of all_sample_files
        assert set(sample_files_config) <= set(all_sample_files), f"Missing files in tarballs: {set(sample_files_config) - set(all_sample_files)}"

    job_str = job_str.format_map({
        'ACCOUNT' : account,
        "USEREMAIL" : email,
        "LOGFILE" : os.path.join(outdir, "slurm-%j.log"),
        "TARBALLLIST" : " ".join(tarballs_set), # remove duplicates
        "RUNCONFIG" : jobcfg_name,
        "INPUTDIR" : inputdir_job,
        "OUTPUTDIR" : outputdir_job,
        "RESULT" : result_tarball
    })

    # write slurm job file
    with open(jobfile_name, 'w') as fout:
        fout.write(job_str)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create Slurm jobs for a given run configuration')

    parser.add_argument('config_name', type=str, help='The run configuration file')
    parser.add_argument('-d', '--sample-dir', type=str, action=util.ParseEnvVar,
                        default="${DATA_DIR}/ntuplerTT/latest",
                        help='Top direcotry for sample files')
    parser.add_argument('-t', '--sample-tarballs', type=str, nargs='+', required=True,
                        help="A list of file paths to input dataset tarballs")
    parser.add_argument('-e', '--email', type=str, action=util.ParseEnvVar, default="${USER}@phas.ubc.ca", help="Email address for job notifications")
    parser.add_argument('-a', '--account', type=str, default="def-alister", help="Slurm account")
    parser.add_argument('-o', '--output-name', type=str, help="Slurm job file name")
    parser.add_argument('--output-dir', type=str, help="Output directory")

    args = parser.parse_args()

    generate_slurm_jobs(**vars(args))