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

args = P.parse_args()

g_cfg = gt.get_gcorr_config(args.gcorr_config)
tags = g_cfg["gcorr_opts"]["map_tags"]
null = g_cfg["gcorr_opts"]["null"]
rundir = g_cfg["gcorr_opts"]["output_root"]

run_opts = dict(
    map_tags=tags,
    data_subset=g_cfg["gcorr_opts"]["data_subset"],
    output_root=rundir,
    null=null,
    **g_cfg["xfaster_opts"],
)

if args.submit:
    run_opts.update(submit=True, wait=True, **g_cfg["submit_opts"])

# If rundir doesn't exist or force_restart, we start from scratch
if not os.path.exists(rundir) or args.force_restart:
    iternum = 0

    for gfile in glob.glob(os.path.join(rundir, "*", "gcorr*.npz")):
        os.remove(gfile)

else:
    gfiles = glob.glob(os.path.join(rundir, "*", "gcorr*_iter*.npz"))
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
    flist = ["{}/{}*".format(rundirf, f) for f in flist]

    if args.keep_iters:
        # Keep iteration output files
        iterdir = os.path.join(rundirf, "iter{:03d}".format(iternum))
        os.mkdir(iterdir)
        for f in flist:
            sp.check_call("rsync -a {} {}/".format(f, iterdir), shell=True)

    # Remove transfer functions and bandpowers from run directory
    for f in flist:
        sp.check_call("rm -rf {}".format(f), shell=True)
