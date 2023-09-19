"""
A script for computing g_corr factor from simulation bandpowers.
"""
import os
import glob
import numpy as np
import scipy.optimize as opt
import argparse as ap
from configparser import ConfigParser
import xfaster as xf
from xfaster import parse_tools as pt

P = ap.ArgumentParser()
P.add_argument("--gcorr-config", help="The config file for gcorr computation")
P.add_argument("--output-tag", help="Which map tag")
P.add_argument("-r", "--root", default="xfaster_gcal", help="XFaster outputs directory")
args = P.parse_args()


assert os.path.exists(args.gcorr_config), "Missing config file {}".format(
    args.gcorr_config
)
g_cfg = ConfigParser()
g_cfg.read(args.gcorr_config)
null = g_cfg.getboolean("gcorr_opts", "null")
if null:
    # no sample variance used for null tests
    fish_name = "invfish_nosampvar"
else:
    fish_name = "inv_fish"

output_tag = args.output_tag
output_root = os.path.join(g_cfg["gcorr_opts"]["output_root"], args.root)

specs = ["tt", "ee", "bb", "te", "eb", "tb"]

nsim = g_cfg.getint("gcorr_opts", "nsim")


# use gauss model for null bandpowers
def gauss(qb, amp, width, offset):
    # width = 0.5*1/sig**2
    # offset = mean
    return amp * np.exp(-width * (qb - offset) ** 2)


# use lognormal model for signal bandpowers
def lognorm(qb, amp, width, offset):
    return gauss(np.log(qb), amp, width, offset)


# Compute the correction factor
if output_root is None:
    output_root = os.getcwd()
if output_tag is not None:
    output_root = os.path.join(output_root, output_tag)
    output_tag = "_{}".format(output_tag)
else:
    output_tag = ""

file_glob = os.path.join(output_root, "bandpowers_sim*{}.npz".format(output_tag))
files = sorted(glob.glob(file_glob))
if not len(files):
    raise OSError("No bandpowers files found in {}".format(output_root))

out = {"data_version": 1}
inv_fishes = None
qbs = {}

for spec in specs:
    qbs[spec] = None

for filename in files:
    bp = xf.load_and_parse(filename)
    inv_fish = bp[fish_name]
    bad = np.where(np.diag(inv_fish) < 0)[0]
    if len(bad):
        # this happens rarely and we won't use those sims
        print("Found negative fisher values in {}: {}".format(filename, bad))
        continue

    if inv_fishes is None:
        inv_fishes = np.diag(inv_fish)
    else:
        inv_fishes = np.vstack([inv_fishes, np.diag(inv_fish)])

    for spec in specs:
        if qbs[spec] is None:
            qbs[spec] = bp["qb"]["cmb_{}".format(spec)]
        else:
            qbs[spec] = np.vstack([qbs[spec], bp["qb"]["cmb_{}".format(spec)]])

# Get average XF-estimated variance
xf_var_mean = np.mean(inv_fishes, axis=0)
xf_var = pt.arr_to_dict(xf_var_mean, bp["qb"])

out["bin_def"] = bp["bin_def"]
nbins = len(out["bin_def"]["cmb_tt"])
out["gcorr"] = {}

for spec in specs:
    stag = "cmb_{}".format(spec)
    out["gcorr"][spec] = np.ones(nbins)
    for b0 in np.arange(nbins):
        hist, bins = np.histogram(
            np.asarray(qbs[spec])[:, b0], density=True, bins=int(nsim / 10.0)
        )
        bc = (bins[:-1] + bins[1:]) / 2.0

        # Gauss Fisher-based params
        A0 = np.max(hist)
        sig0 = np.sqrt(xf_var[stag][b0])
        mu0 = np.mean(qbs[spec][b0])

        if spec in ["eb", "tb"] or null:
            func = gauss
        else:
            func = lognorm
            sig0 /= mu0
            mu0 = np.log(mu0)

        # Initial parameter guesses
        p0 = [A0, 1.0 / sig0**2 / 2.0, mu0]

        try:
            popth, pcovh = opt.curve_fit(func, bc, hist, p0=p0, maxfev=int(1e9))
            # gcorr is XF Fisher variance over fit variance
            out["gcorr"][spec][b0] = popth[1] / p0[1]
        except RuntimeError:
            print("No hist fits found")

outfile = os.path.join(output_root, "gcorr_corr{}.npz".format(output_tag))
np.savez_compressed(outfile, **out)
print("New gcorr correction computed (should converge to 1): ", out["gcorr"])
