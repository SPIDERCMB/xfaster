"""
A script to run XFaster for gcorr calculation. Called by iterate.py.
"""
import argparse as ap

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

from xfaster.gcorr_tools import run_xfaster_gcorr

run_xfaster_gcorr(
    args.gcorr_config,
    output=args.output,
    apply_gcorr=args.gcorr,
    reload_gcorr=args.reload_gcorr,
    checkpoint=args.check_point,
    submit=args.submit,
    omp_threads=args.omp,
    first_seed=args.first,
    num_seeds=args.num,
)
