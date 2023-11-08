"""
A iteration script used to update g_corr factors. Can either be run in series
or submit jobs to run in parallel.
"""
import os
import argparse as ap
from xfaster import gcorr_tools as gt
from xfaster.batch_tools import batch_sub
from matplotlib import use

use("agg")

P = ap.ArgumentParser()
P.add_argument(
    "--gcorr-config", required=True, help="The config file for gcorr computation"
)
P.add_argument(
    "-n",
    "--no-submit",
    dest="submit",
    action="store_false",
    help="Don't submit jobs; run serially on current session",
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

args = P.parse_args()

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
    for k in ["allow_extreme", "gcorr_fit_hist", "keep_iters", "force_restart"]:
        if getattr(args, k):
            cmd += ["--{}".format(k.replace("_", "-"))]

# run
for tag in tags:
    if not args.analyze_only:
        jobs = gt.xfaster_gcorr(output_tag=tag, **run_opts)

    if args.submit:
        batch_sub(
            cmd + ["-n", "-a", "-t", tag], dep_afterok=jobs, **g_cfg["submit_opts"]
        )
    else:
        gt.process_gcorr(output_tag=tag, **gcorr_opts)
