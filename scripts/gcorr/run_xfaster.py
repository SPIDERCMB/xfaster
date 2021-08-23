"""
A script to run XFaster for gcorr calculation. Called by iterate.py.
"""
import os
import xfaster as xf
import argparse as ap
from configparser import ConfigParser

# Change XFaster options here to suit your purposes
opts = dict(
    likelihood=False,
    residual_fit=False,
    foreground_fit=False,
    # change options below for your purposes
    tbeb=True,
    bin_width=25,
    lmin=2,
    lmax=500,
)

# Change submit options here to fit your system
submit_opts = dict(nodes=1, ppn=1, mem=6, omp_threads=10, wallt=4)

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
P.add_argument(
    "--omp",
    default=None,
    type=int,
    help="Number of omp threads, if submit. Overwrites value in config file",
)

args = P.parse_args()

# start by loading up gcorr config file and parsing it
assert os.path.exists(args.gcorr_config), "Missing config file {}".format(
    args.gcorr_config
)
g_cfg = ConfigParser()
g_cfg.read(args.gcorr_config)

# set all user-specific xfaster opts
for k, v in g_cfg["xfaster_opts"].items():
    opts[k] = v
null = g_cfg.getboolean("gcorr_opts", "null")
tags = g_cfg["gcorr_opts"]["map_tags"].split(",")

# null tests should use noise sims. signal shouldn't.
if null:
    opts["noise_type"] = g_cfg["xfaster_opts"]["noise_type"]
    opts["sim_data_components"] = ["signal", "noise"]
else:
    opts["noise_type"] = None
    opts["sim_data_components"] = ["signal"]

opts["output_root"] = os.path.join(g_cfg["gcorr_opts"]["output_root"], args.output)

# update opts with command line args
opts["apply_gcorr"] = args.gcorr
opts["reload_gcorr"] = args.reload_gcorr
opts["checkpoint"] = args.check_point

seeds = list(range(args.first, args.first + args.num))

for tag in tags:
    opts["sim_data"] = True
    opts["output_tag"] = tag
    opts["gcorr_file"] = os.path.abspath(
        os.path.join(
            g_cfg["gcorr_opts"]["output_root"],
            "xfaster_gcal",
            tag,
            "gcorr_{}_total.npz".format(tag),
        )
    )
    opts["data_subset"] = os.path.join(
        g_cfg["gcorr_opts"]["data_subset"], "*{}".format(tag)
    )
    if args.omp is not None:
        submit_opts["omp_threads"] = args.omp

    if args.submit:
        opts.update(**submit_opts)

    for s in seeds:
        opts["sim_index_default"] = s
        if args.submit:
            xf.xfaster_submit(**opts)
        else:
            xf.xfaster_run(**opts)
