"""
A iteration script used to update g_corr factors.
"""
import os
import numpy as np
import argparse as ap
from configparser import ConfigParser
import time
import copy
import glob
import shutil
from matplotlib import use

use("agg")
import matplotlib.pyplot as plt
import xfaster as xf
from xfaster import parse_tools as pt

P = ap.ArgumentParser()
P.add_argument("--gcorr-config", help="The config file for gcorr computation")
P.add_argument(
    "--force-restart",
    action="store_true",
    default=False,
    help="Force restarting from iteration 0-- remakes iteration dir",
)

args = P.parse_args()

assert os.path.exists(args.gcorr_config), "Missing config file {}".format(
    args.gcorr_config
)
g_cfg = ConfigParser()
g_cfg.read(args.gcorr_config)
tags = g_cfg["gcorr_opts"]["map_tags"].split(",")

specs = ["tt", "ee", "bb", "te", "eb", "tb"]

run_name = "xfaster_gcal"
run_name_iter = run_name + "_iter"
# ref dir will contain the total gcorr file
ref_dir = os.path.join(g_cfg["gcorr_opts"]["output_root"], run_name)
# run dir will be where all the iteration happens to update the reference
rundir = ref_dir + "_iter"

# If we've just started iterating, create the iterating directory
if os.getenv("GCORR_ITER") in [None, "0"] or args.force_restart:
    if os.path.exists(rundir):
        shutil.rmtree(rundir)
    for gfile in glob.glob(os.path.join("ref_dir", "*", "gcorr*.npz")):
        os.remove(gfile)

    shutil.copytree(ref_dir, rundir)
    os.environ["GCORR_ITER"] = "0"

    # make plots directory
    os.mkdir(os.path.join(rundir, "plots"))

    # Since this is the first iteration, we also make a gcorr file of
    # all ones.
    for tag in tags:
        bp_file = glob.glob(os.path.join(ref_dir, tag, "bandpowers*.npz"))[0]
        bp = xf.load_and_parse(bp_file)
        gcorr_data = {"bin_def": bp["bin_def"], "gcorr": {}, "data_version": 1}
        for spec in specs:
            stag = "cmb_{}".format(spec)
            gcorr_data["gcorr"][spec] = np.ones_like(bp["qb"][stag])
        np.savez_compressed(
            os.path.join(ref_dir, tag, "gcorr_{}_total.npz".format(tag)), **gcorr_data
        )

else:
    os.environ["GCORR_ITER"] = str(int(os.environ["GCORR_ITER"]) + 1)
print("Starting iteration {}".format(os.environ["GCORR_ITER"]))


for tag in tags:
    ref_file = os.path.join(ref_dir, "{0}/gcorr_{0}_total.npz".format(tag))
    rundirf = os.path.join(rundir, tag)

    # Remove transfer functions and bandpowers
    os.system("rm -rf {}/bandpowers*".format(rundirf))
    os.system("rm -rf {}/transfer*".format(rundirf))
    os.system("rm -rf {}/ERROR*".format(rundirf))
    os.system("rm -rf {}/logs".format(rundirf))

    first = False

    # Get gcorr from reference folder, if it's there
    assert os.path.exists(ref_file), "Missing total gcorr file {}".format(ref_file)
    gcorr = xf.load_and_parse(ref_file)
    print("loaded {}".format(ref_file))

    # Get gcorr_correction from iter folder -- this is the multiplicative
    # change to gcorr-- should converge to 1s
    try:
        gcorr_corr = xf.load_and_parse(
            os.path.join(rundirf, "gcorr_corr_{}.npz".format(tag))
        )
        print(
            "got correction to gcorr {}".format(
                os.path.join(rundirf, "gcorr_gcorr_{}.npz".format(tag))
            )
        )
    except IOError:
        gcorr_corr = copy.deepcopy(gcorr)
        gcorr_corr["gcorr"] = pt.arr_to_dict(
            np.ones_like(pt.dict_to_arr(gcorr["gcorr"], flatten=True)),
            gcorr["gcorr"],
        )
        first = True
        print("Didn't get gcorr correction file in iter folder. Starting from ones.")

    np.savez_compressed(ref_file.replace("_total.npz", "_prev.npz"), **gcorr)

    gcorr["gcorr"] = pt.arr_to_dict(
        pt.dict_to_arr(gcorr["gcorr"], flatten=True)
        * pt.dict_to_arr(gcorr_corr["gcorr"], flatten=True),
        gcorr["gcorr"],
    )

    fig_tot, ax_tot = plt.subplots(2, 3)
    fig_corr, ax_corr = plt.subplots(2, 3)
    ax_tot = ax_tot.flatten()
    ax_corr = ax_corr.flatten()
    for i, (k, v) in enumerate(gcorr["gcorr"].items()):
        v[0] = 0.5
        if k in ["te", "eb", "tb"]:
            # We don't compute gcorr for off-diagonals
            v[:] = 1
        # Don't update gcorr if correction is extreme
        v[v < 0.05] /= gcorr_corr["gcorr"][k][v < 0.05]
        v[v > 5] /= gcorr_corr["gcorr"][k][v > 5]
        for v0, val in enumerate(v):
            if val > 1.2:
                if v0 != 0:
                    v[v0] = v[v0 - 1]
                else:
                    v[v0] = 1.2
        ax_tot[i].plot(v)
        ax_tot[i].set_title("{} total gcorr".format(k))
        ax_corr[i].plot(gcorr_corr["gcorr"][k])
        ax_tot[i].set_title("{} gcorr corr".format(k))

    print(gcorr["gcorr"])
    fig_tot.savefig(
        os.path.join(
            rundir,
            "plots",
            "gcorr_tot_{}_iter{}.png".format(tag, os.getenv("GCORR_ITER")),
        )
    )
    fig_corr.savefig(
        os.path.join(
            rundir,
            "plots",
            "gcorr_corr_{}_iter{}.png".format(tag, os.getenv("GCORR_ITER")),
        )
    )

    np.savez_compressed(ref_file, **gcorr)

print("Sumitting first job")
print(
    "python run_xfaster.py --gcorr-config {g} --omp 1 --check-point transfer --reload-gcorr -o {o} > /dev/null".format(
        g=args.gcorr_config, o=run_name_iter
    )
)
os.system(
    "python run_xfaster.py --gcorr-config {g} --omp 1 --check-point transfer --reload-gcorr -o {o} > /dev/null".format(
        g=args.gcorr_config, o=run_name_iter
    )
)

transfer_exists = {}
for tag in tags:
    transfer_exists[tag] = False
print(transfer_exists)
for v in transfer_exists.values():
    print("transfer", v)
print("all", np.all(list(transfer_exists.values())))
while not np.all(list(transfer_exists.values())):
    print("waiting for transfer to exists")
    # wait until transfer functions are done to submit rest of jobs
    for tag in tags:
        rundirf = os.path.join(rundir, tag)
        transfer_files = glob.glob(
            os.path.join(rundirf, "transfer_all*{}.npz".format(tag))
        )
        transfer_exists[tag] = bool(len(transfer_files))

    print("transfer exists: ", transfer_exists)
    time.sleep(15)
print("Submitting jobs for all seeds")
print(
    "python run_xfaster.py --gcorr-config {g} --omp 1 --check-point bandpowers -o {o} -f 1 -n {n} > /dev/null".format(
        g=args.gcorr_config, o=run_name_iter, n=g_cfg["gcorr_opts"]["nsim"]
    )
)

os.system(
    "python run_xfaster.py --gcorr-config {g} --omp 1 --check-point bandpowers -o {o} -f 1 -n {n} > /dev/null".format(
        g=args.gcorr_config, o=run_name_iter, n=g_cfg["gcorr_opts"]["nsim"]
    )
)

print("Waiting for jobs to complete...")
while os.system("squeue -u {} | grep xfast > /dev/null".format(os.getenv("USER"))) == 0:
    os.system("squeue -u {} | wc".format(os.getenv("USER")))
    time.sleep(10)

for tag in tags:
    os.system(
        "python compute_gcal.py --gcorr-config {g} -r {r} {t}".format(
            g=args.gcorr_config, r=run_name_iter, t=tag
        )
    )

for tag in tags:
    print("{} completed".format(tag))
    os.system("ls {}/{}/bandpowers* | wc -l".format(rundir, tag))
    print("{} error".format(tag))
    os.system("ls {}/{}/ERROR* | wc -l".format(rundir, tag))
