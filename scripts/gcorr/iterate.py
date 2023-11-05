"""
A iteration script used to update g_corr factors. Can either be run in series
or submit jobs to run in parallel.
"""
import os
import argparse as ap
import glob
import shutil
import subprocess as sp
from xfaster import gcorr_tools as gt
from matplotlib import use

use("agg")

P = ap.ArgumentParser()
P.add_argument("--gcorr-config", help="The config file for gcorr computation")
P.add_argument(
    "--no-submit",
    dest="submit",
    action="store_false",
    help="Don't submit jobs; run serially on current session",
)
P.add_argument(
    "--force-restart",
    action="store_true",
    default=False,
    help="Force restarting from iteration 0-- remakes iteration dir",
)
P.add_argument(
    "--allow-extreme",
    action="store_true",
    help="Do not clip gcorr corrections that are too large at each "
    "iteration.  Try this if iterations are not converging.",
)
P.add_argument(
    "--gcorr-fit-hist",
    action="store_true",
    help="Fit bandpower histogram to a lognorm distribution to compute gcorr",
)
P.add_argument(
    "--keep-iters",
    action="store_true",
    help="Store outputs from each iteration in a separate directory",
)
P.add_argument(
    "--reference",
    action="store_true",
    help="Run xfaster to build or rebuild the reference directory. "
    "Implies --force-restart as well.",
)

args = P.parse_args()

g_cfg = gt.get_gcorr_config(args.gcorr_config)
tags = g_cfg["gcorr_opts"]["map_tags"]
null = g_cfg["gcorr_opts"]["null"]

run_name = "xfaster_gcal"
run_name_iter = run_name + "_iter"
# ref dir will contain the total gcorr file
ref_dir = os.path.join(g_cfg["gcorr_opts"]["output_root"], run_name)
# run dir will be where all the iteration happens to update the reference
rundir = ref_dir + "_iter"

run_opts = dict(
    map_tags=tags,
    data_subset=g_cfg["gcorr_opts"]["data_subset"],
    output_root=rundir,
    null=null,
    **g_cfg["xfaster_opts"],
)

if args.submit:
    run_opts.update(submit=True, wait=True, **g_cfg["submit_opts"])

# Submit an initial job that computes all of the signal and noise spectra that
# you will need to run the subsequent gcorr iterations.  This job should use as
# many omp_threads as possible.
if args.reference or not os.path.exists(ref_dir):
    ref_opts = run_opts.copy()
    ref_opts["output_root"] = ref_dir
    print("Generating reference run {}".format(ref_dir))
    args.force_restart = True
    gt.run_xfaster_gcorr(apply_gcorr=False, **ref_opts)

# If rundir doesn't exist or force_restart, we start from scratch
if not os.path.exists(rundir) or args.force_restart:
    iternum = 0

    if os.path.exists(rundir):
        shutil.rmtree(rundir)
    for gfile in glob.glob(os.path.join(ref_dir, "*", "gcorr*.npz")):
        os.remove(gfile)
    shutil.copytree(ref_dir, rundir)

else:
    gfiles = glob.glob(os.path.join(ref_dir, "*", "gcorr*_iter*.npz"))
    if not len(gfiles):
        iternum = 0
    else:
        gfiles = [os.path.basename(f).split(".")[0] for f in gfiles]
        iternum = max([int(f.split("iter")[-1]) for f in gfiles]) + 1
print("Starting iteration {}".format(iternum))

# Submit a first job that reloads gcorr and computes the transfer function that
# will be used by all the other seeds
print("Submitting first job")
gt.run_xfaster_gcorr(
    checkpoint="transfer", apply_gcorr=(iternum > 0), reload_gcorr=True, **run_opts
)

# Make sure all transfer functions have been computed
for tag in tags:
    tf = os.path.join(rundir, tag, "transfer_all*{}.npz".format(tag))
    if not len(glob.glob(tf)):
        raise RuntimeError("Missing transfer functions for tag {}".format(tag))

# Once transfer function is done, all other seeds can run
print("Submitting jobs for all seeds")
gt.run_xfaster_gcorr(
    checkpoint="bandpowers",
    apply_gcorr=(iternum > 0),
    sim_index=1,
    num_sims=int(g_cfg["gcorr_opts"]["nsim"]) - 1,
    **run_opts,
)

print("Computing new gcorr factors")
for tag in tags:
    # Compute gcorr correction from bandpower distribution
    out = gt.compute_gcal(
        output_root=rundir,
        output_tag=tag,
        null=null,
        fit_hist=args.gcorr_fit_hist,
    )
    print("New gcorr correction computed (should converge to 1): ", out["gcorr"])

    # Apply correction to cumulative gcorr
    gcorr = gt.apply_gcal(
        output_root=rundir,
        output_tag=tag,
        iternum=iternum,
        allow_extreme=args.allow_extreme,
    )
    print("Total gcorr {}: {}".format(tag, gcorr["gcorr"]))

# Cleanup output directories
for tag in tags:
    rundirf = os.path.join(rundir, tag)
    bp_files = glob.glob(os.path.join(rundirf, "bandpowers*"))
    error_files = glob.glob(os.path.join(rundirf, "ERROR*"))
    print("{} completed: {}".format(tag, len(bp_files)))
    print("{} error: {}".format(tag, len(error_files)))

    flist = ["bandpowers", "transfer", "logs"]
    if len(error_files) > 0:
        flist += ["ERROR"]

    if args.keep_iters:
        # Keep iteration output files
        rundirf_iter = os.path.join(rundirf, "iter{:03d}".format(iternum))
        os.mkdir(rundirf_iter)
        for f in flist + ["gcorr"]:
            sp.check_call(
                "rsync -a {}/{}* {}/".format(rundirf, f, rundirf_iter), shell=True
            )

    # Keep gcorr iteration files
    for f in glob.glob("{}/gcorr*.npz".format(rundirf)):
        rf = f.replace(".npz", "_iter{:03d}.npz".format(iternum))
        rf = rf.replace(rundirf, "{}/{}".format(ref_dir, tag))
        sp.check_call("rsync -a {} {}".format(f, rf), shell=True)

    # Remove transfer functions and bandpowers from run directory
    for f in flist:
        sp.check_call("rm -rf {}/{}*".format(rundirf, f), shell=True)
