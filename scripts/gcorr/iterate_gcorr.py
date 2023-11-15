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
P.add_argument("config", help="The config file for gcorr computation")
P.add_argument(
    "-s",
    "--submit",
    action="store_true",
    help="Submit jobs, instead of running serially on current session",
)
P.add_argument(
    "-t",
    "--map-tags",
    nargs="+",
    help="Subset of map tags to iterate over",
)
P.add_argument(
    "-i",
    "--iternums",
    type=int,
    nargs="+",
    help="Iteration number to start with for each tag in --map-tags.  If one "
    "number is given, use the same index for all tags.  All files for iterations "
    "above this number will be removed.  If not supplied, iterations will increment "
    "to the next index.",
)
P.add_argument(
    "--keep-iters",
    action="store_true",
    help="Store outputs from each iteration in a separate directory",
)
P.add_argument(
    "--max-iters",
    default=0,
    type=int,
    help="Maximum number of iterations to run.  If 0, run once and exit.",
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
    "-a",
    "--analyze-only",
    action="store_true",
    help="Compute and store gcorr files for the current iteration.  Typically used "
    "internally by the script, and should not be called by the user.",
)

args = P.parse_args()

# load configuration
g_cfg = gt.get_gcorr_config(args.config)

tags = g_cfg["gcorr_opts"]["map_tags"]
if args.map_tags:
    tags = [t for t in args.map_tags if t in tags]

iternums = {t: None for t in tags}
if args.iternums:
    if len(args.iternums) == 1:
        args.iternums = args.iternums * len(tags)
    for t, i in zip(tags, args.iternums):
        iternums[t] = i

null = g_cfg["gcorr_opts"].get("null", False)
nsim = g_cfg["gcorr_opts"]["nsim"]
rundir = g_cfg["gcorr_opts"]["output_root"]
xf_submit_opts = g_cfg.get("submit_opts", {})
submit_opts = xf_submit_opts.copy()
submit_opts.pop("num_jobs")

# sim ensemble options
run_opts = dict(
    data_subset=g_cfg["gcorr_opts"]["data_subset"],
    data_subset2=g_cfg["gcorr_opts"].get("data_subset2", None),
    output_root=rundir,
    null=null,
    num_sims=nsim,
    **g_cfg["xfaster_opts"],
)
if args.submit:
    run_opts.update(submit=True, **xf_submit_opts)

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

if args.analyze_only:
    assert len(tags) == 1, "Analyze one tag at a time"
    if gt.process_gcorr(output_tag=tags[0], **gcorr_opts):
        raise RuntimeError("Stopping iterations")
    raise SystemExit

# build command for this script
cmd = ["python", os.path.abspath(__file__), os.path.abspath(args.config)]
for k in [
    "allow_extreme",
    "gcorr_fit_hist",
    "keep_iters",
]:
    if getattr(args, k):
        cmd += ["--{}".format(k.replace("_", "-"))]
for k in ["converge_criteria", "max_iters"]:
    cmd += ["--{}".format(k.replace("_", "-")), str(getattr(args, k))]

if args.max_iters == 0:
    args.max_iters = 1

# run
for tag in tags:
    # setup for next iteration
    iternum0 = gt.get_next_iter(
        output_root=rundir, output_tag=tag, iternum=iternums[tag]
    )
    if iternum0 > args.max_iters:
        raise ValueError(
            "Tag {} iteration {} > max {}".format(tag, iternum0, args.max_iters)
        )
    gcorr_job = None

    for iternum in range(iternum0, args.max_iters):
        print("Starting {} iteration {}".format(tag, iternum))

        # compute ensemble bandpowers
        if args.submit:
            run_opts["dep_afterok"] = gcorr_job
        bp_jobs = gt.xfaster_gcorr(output_tag=tag, **run_opts)

        # compute gcorr
        if args.submit:
            # submit analysis job
            gcorr_job = batch_sub(
                cmd + ["-a", "-t", tag],
                name="gcorr_{}".format(tag),
                workdir=os.path.abspath(os.path.join(rundir, tag, "logs")),
                dep_afterok=bp_jobs,
                **submit_opts,
            )
        else:
            if gt.process_gcorr(output_tag=tag, **gcorr_opts):
                break
