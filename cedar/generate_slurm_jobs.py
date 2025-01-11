import os
import util

def generate_slurm_jobs(
    config_name, # file name of the run config
    sample_dir, # top direcotry for sample files
    sample_tarballs, # a list of file paths to input dataset tarballs
    email = os.getenv('USER')+'@phas.ubc.ca', # for now
    output_name = None, # slurm job file name,
    ):

    if not output_name:
        # set it to the same as config_name but with a different extension
        output_name = os.path.splitext(config_name)[0]+'.slurm'

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

    # overwrite the old run config file
    util.write_dict_to_json(runcfg, config_name)

    # Load slurm job template
    slurm_template = os.path.expandvars("${SOURCE_DIR}/cedar/slurmJob.template")
    with open(slurm_template) as ftmp:
        job_str = ftmp.read()

    job_str = job_str.format_map({
        "USEREMAIL" : email,
        "TARBALLLIST" : " ".join(set(sample_tarballs)), # remove duplicates
        "RUNCONFIG" : config_name,
        "INPUTDIR" : inputdir_job,
        "OUTPUTDIR" : outputdir_job,
        "RESULT" : result_tarball
    })

    # write slurm job file
    with open(output_name, 'w') as fout:
        fout.write(job_str)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create Slurm jobs for a given run configuration')

    parser.add_argument('config_name', type=str, help='The run configuration file')
    parser.add_argument('-d', '--sample-dir', type=str, action=util.ParseEnvVar,
                        default="${DATA_DIR}/NtupleTT/latest",
                        help='Top direcotry for sample files')
    parser.add_argument('-t', '--sample-tarballs', type=str, nargs='+', required=True,
                        help="A list of file paths to input dataset tarballs")
    parser.add_argument('-e', '--email', type=str, action=util.ParseEnvVar, default="${USER}@phas.ubc.ca", help="Email address for job notifications")
    parser.add_argument('-o', '--output-name', type=str, help="Slurm job file name")

    args = parser.parse_args()

    generate_slurm_jobs(**vars(args))