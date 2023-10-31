"""
A script for computing g_corr factor from simulation bandpowers.
"""
import argparse as ap

P = ap.ArgumentParser()
P.add_argument("--gcorr-config", help="The config file for gcorr computation")
P.add_argument("--output-tag", help="Which map tag")
P.add_argument("-r", "--root", default="xfaster_gcal", help="XFaster outputs directory")
P.add_argument(
    "--fit-hist",
    action="store_true",
    help="Fit histogram for gcorr, rather than using the distribution variance",
)
args = P.parse_args()


from xfaster.gcorr_tools import compute_gcal

compute_gcal(
    args.gcorr_config,
    output=args.root,
    output_tag=args.output_tag,
    fit_hist=args.fit_hist,
)
