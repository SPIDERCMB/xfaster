"""
A iteration script used to update g_corr factors. Can either be run in series
or submit jobs to run in parallel.
"""
import os
import argparse as ap
import subprocess as sp
from xfaster import gcorr_tools as gt
from xfaster.batch_tools import batch_sub
from matplotlib import use

use("agg")

P = ap.ArgumentParser()
P.add_argument(
    "--gcorr-config", required=True, help="The config file for gcorr computation"
)
P.add_argument(
    "-s",
    "--submit",
    action="store_true",
    help="Submit jobs, instead of running serially on current session",
)
P.add_argument(
    "--force-restart",
    action="store_true",
    default=False,
    help="Force restarting from iteration 0",
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
P.add_argument(
    "-a",
    "--analyze-only",
    action="store_true",
    help="Compute and update gcorr files for the current iteration",
)
P.add_argument(
    "-c",
    "--converge-criteria",
    type=float,
    default=0.01,
    help="Maximum fractional change in gcorr that indicates "
    "convergence and stops iteration",
)
P.add_argument(
    "--max-iters",
    default=0,
    type=int,
    help="Maximum number of iterations to run.  If 0, run once and exit.",
)
P.add_argument(
    "--submit-next",
    action="store_true",
    default=None,
    help="Submit jobs for the next iteration (rather than running them locally). "
    "This option is used internally by the script, do not set this option manually.",
)

args = P.parse_args()

if args.submit_next is None:
    args.submit_next = args.submit

# load configuration
g_cfg = gt.get_gcorr_config(args.gcorr_config)

tags = g_cfg["gcorr_opts"]["map_tags"]
if args.map_tags:
    tags = [t for t in args.map_tags if t in tags]

null = g_cfg["gcorr_opts"]["null"]
nsim = g_cfg["gcorr_opts"]["nsim"]
rundir = g_cfg["gcorr_opts"]["output_root"]

# sim ensemble options
run_opts = dict(
    force_restart=args.force_restart,
    data_subset=g_cfg["gcorr_opts"]["data_subset"],
    output_root=rundir,
    null=null,
    num_sims=nsim,
    **g_cfg["xfaster_opts"],
)

# gcorr analysis options
gcorr_opts = dict(
    output_root=rundir,
    null=null,
    num_sims=nsim,
    gcorr_fit_hist=args.gcorr_fit_hist,
    allow_extreme=args.allow_extreme,
    keep_iters=args.keep_iters,
    converge_criteria=args.converge_criteria,
    max_iters=args.max_iters,
)

# build command for this script
if args.submit:
    run_opts.update(submit=True, **g_cfg["submit_opts"])

cmd = [
    "python",
    os.path.abspath(__file__),
    "--gcorr-config",
    os.path.abspath(args.gcorr_config),
]
for k in [
    "allow_extreme",
    "gcorr_fit_hist",
    "keep_iters",
    "force_restart",
    "submit_next",
]:
    if getattr(args, k):
        cmd += ["--{}".format(k.replace("_", "-"))]
for k in ["converge_criteria", "max_iters"]:
    cmd += ["--{}".format(k.replace("_", "-")), str(getattr(args, k))]

# run
for tag in tags:
    if not args.analyze_only:
        jobs = gt.xfaster_gcorr(output_tag=tag, **run_opts)

    tag_cmd = cmd + ["-t", tag]

    if args.submit:
        # submit analysis job
        batch_sub(tag_cmd + ["-a"], dep_afterok=jobs, **g_cfg["submit_opts"])
    else:
        if not gt.process_gcorr(output_tag=tag, **gcorr_opts):
            # run again if not converged or reached max_iters
            sp.check_call(tag_cmd + ["-s"] * args.submit_next)
