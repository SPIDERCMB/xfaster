"""
A script to run XFaster for gcorr calculation. Called by iterate.py.
"""
import os
import xfaster as xf
import argparse as ap
from configparser import ConfigParser

P = ap.ArgumentParser()
P.add_argument("--gcorr-config", help="The config file for gcorr computation")
P.add_argument("-f", "--first", default=0, type=int, help="First sim index to run")
P.add_argument("-n", "--num", default=1, type=int, help="Number of sims to run")
P.add_argument(
    "-o", "--output", default="xfaster_gcal", help="Name of output subdirectory"
)
P.add_argument(
    "--no-gcorr",
    dest="gcorr",
    default=True,
    action="store_false",
    help="Don't apply a g-gcorrection",
)
P.add_argument(
    "--reload-gcorr", default=False, action="store_true", help="Reload the gcorr factor"
)
P.add_argument("--check-point", default="bandpowers", help="XFaster checkpoint")
P.add_argument(
    "--no-submit", dest="submit", action="store_false", help="Don't submit, run locally"
)
P.add_argument("--omp", default=None, type=int, help="Number of omp threads, if submit. Overwrites value in config file")

output_root = "../../example/gcorr_run/"  # Set your own output root.

args = P.parse_args()

assert os.path.exists(args.gcorr_config), "Missing config file {}".format(
    args.gcorr_config
)
g_cfg = ConfigParser()
g_cfg.read(args.gcorr_config)
null = g_cfg.getboolean("gcorr_opts", "null")

seeds = list(range(args.first, args.first + args.num))

tags = g_cfg["gcorr_opts"]["map_tags"]

for tag in tags:
    opts = dict(
        output_root=os.path.join(g_cfg["gcorr_opts"]["output_root"], args.output),
        output_tag=tag,
        likelihood=False,
        residual_fit=False,
        foreground_fit=False,
        apply_gcorr=args.gcorr,
        reload_gcorr=args.reload_gcorr,
        checkpoint=args.check_point,
    )
    # set all user-specific xfaster opts
    for k, v in g_cfg["xfaster_opts"].items():
        opts[k] = v

    # null tests should use noise sims. signal shouldn't.
    if null:
        opts["noise_type"] = g_cfg["xfaster_opts"]["noise_type"]
    else:
        opts["noise_type"] = None

    submit_opts = g_cfg["submit_opts"]
    if args.omp is not None:
        submit_opts.update("omp_threads", args.omp)

    if args.submit:
        opts.update(**submit_opts)

    for s in seeds:
        opts.update(sim_index=s)
        if args.submit:
            xf.xfaster_submit(**opts)
        else:
            xf.xfaster_run(**opts)
