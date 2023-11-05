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
    "-t",
    "--map-tags",
    nargs="+",
    help="Subset of map tags to iterate over",
)

args = P.parse_args()

g_cfg = gt.get_gcorr_config(args.gcorr_config)

tags = g_cfg["gcorr_opts"]["map_tags"]
if args.map_tags:
    tags = [t for t in args.map_tags if t in tags]

null = g_cfg["gcorr_opts"]["null"]
rundir = g_cfg["gcorr_opts"]["output_root"]

run_opts = dict(
    data_subset=g_cfg["gcorr_opts"]["data_subset"],
    output_root=rundir,
    null=null,
    **g_cfg["xfaster_opts"],
)

if args.submit:
    run_opts.update(submit=True, **g_cfg["submit_opts"])

iternum = {tag: 0 for tag in tags}

# If rundir doesn't exist or force_restart, we start from scratch
if not os.path.exists(rundir) or args.force_restart:
    for gfile in glob.glob(os.path.join(rundir, "*", "gcorr*.npz")):
        os.remove(gfile)
else:
    for tag in tags:
        gfiles = glob.glob(
            os.path.join(rundir, "*", "gcorr_total_{}_iter*.npz".format(tag))
        )
        iternum[tag] = len(gfiles)
print("Starting iteration {}".format(iternum))

# Submit a first job that reloads gcorr and computes the transfer function that
# will be used by all the other seeds
print("Submitting first jobs")
transfer_jobs = []
for tag in tags:
    jobs = gt.run_xfaster_gcorr(
        output_tag=tag,
        checkpoint="transfer",
        apply_gcorr=(iternum[tag] > 0),
        reload_gcorr=True,
        **run_opts,
    )
    transfer_jobs.extend(jobs)

# Make sure all transfer functions have been computed
if args.submit:
    print("Waiting for transfer function jobs to complete")
    gt.wait_for_jobs(transfer_jobs)
for tag in tags:
    tf = os.path.join(rundir, tag, "transfer_all*{}.npz".format(tag))
    if not len(glob.glob(tf)):
        raise RuntimeError("Missing transfer functions for tag {}".format(tag))

# Once transfer function is done, all other seeds can run
print("Submitting jobs for all seeds")
sim_jobs = []
for tag in tags:
    jobs = gt.run_xfaster_gcorr(
        output_tag=tag,
        checkpoint="bandpowers",
        apply_gcorr=(iternum[tag] > 0),
        sim_index=1,
        num_sims=int(g_cfg["gcorr_opts"]["nsim"]) - 1,
        **run_opts,
    )
    sim_jobs.extend(jobs)
if args.submit:
    print("Waiting for sim ensemble jobs to complete")
    gt.wait_for_jobs(sim_jobs)

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
        iternum=iternum[tag],
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
        iterdir = os.path.join(rundirf, "iter{:03d}".format(iternum[tag]))
        os.mkdir(iterdir)
        for f in flist:
            sp.check_call("rsync -a {} {}/".format(f, iterdir), shell=True)

    # Remove transfer functions and bandpowers from run directory
    for f in flist:
        sp.check_call("rm -rf {}".format(f), shell=True)
