"""
A iteration script used to update g_corr factors. Can either be run in series
or submit jobs to run in parallel.
"""
import os
import numpy as np
import argparse as ap
import time
import copy
import glob
import shutil
import subprocess as sp
from matplotlib import use

use("agg")
import matplotlib.pyplot as plt
import xfaster as xf
from xfaster import parse_tools as pt
from xfaster import gcorr_tools as gt

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
    "--keep-iters",
    action="store_true",
    help="Store outputs from each iteration in a separate directory",
)
P.add_argument(
    "--reference",
    action="store_true",
    help="Run xfaster to build or rebuild the reference directory. "
    "Implies --force-restart as well.",
)

args = P.parse_args()

g_cfg = gt.get_gcorr_config(args.gcorr_config)
tags = g_cfg["gcorr_opts"]["map_tags"]

specs = ["tt", "ee", "bb", "te", "eb", "tb"]

run_name = "xfaster_gcal"
run_name_iter = run_name + "_iter"
# ref dir will contain the total gcorr file
ref_dir = os.path.join(g_cfg["gcorr_opts"]["output_root"], run_name)
# run dir will be where all the iteration happens to update the reference
rundir = ref_dir + "_iter"

run_opts = dict(
    cfg=g_cfg,
    output=run_name_iter,
    submit=args.submit,
)

# Submit an initial job that computes the all of the signal and noise spectra
# that you will need to run the subsequent gcorr iterations.  This job should
# use as many omp_threads as possible.
if args.reference or not os.path.exists(ref_dir):
    ref_opts = run_opts.copy()
    ref_opts["output"] = run_name
    print("Generating reference run {}".format(ref_dir))
    args.force_restart = True
    gt.run_xfaster_gcorr(apply_gcorr=False, **ref_opts)
    if args.submit:
        print("Wait until reference run completes, then run this script again")
        raise SystemExit

# if rundir doesn't exist or force_restart, we start from scratch
if not os.path.exists(rundir) or args.force_restart:
    iternum = 0

    if os.path.exists(rundir):
        shutil.rmtree(rundir)
    for gfile in glob.glob(os.path.join(ref_dir, "*", "gcorr*.npz")):
        os.remove(gfile)
    shutil.copytree(ref_dir, rundir)

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
    # check plots directory to find what iteration we're at
    plots = glob.glob(os.path.join(rundir, "plots", "*.png"))
    plot_inds = sorted([int(x.split("iter")[-1].split(".")[0]) for x in plots])
    last_iter = plot_inds[-1]
    iternum = int(last_iter) + 1
print("Starting iteration {}".format(iternum))


for tag in tags:
    ref_file = os.path.join(ref_dir, "{0}/gcorr_{0}_total.npz".format(tag))
    rundirf = os.path.join(rundir, tag)

    first = False

    # Get gcorr from reference folder, if it's there
    assert os.path.exists(ref_file), "Missing total gcorr file {}".format(ref_file)
    gcorr = xf.load_and_parse(ref_file)
    print("loaded {}".format(ref_file))

    # Get gcorr_correction from iter folder -- this is the multiplicative
    # change to gcorr-- should converge to 1s
    try:
        fp = os.path.join(rundirf, "gcorr_corr_{}.npz".format(tag))
        gcorr_corr = xf.load_and_parse(fp)
        print("got correction to gcorr {}".format(fp))
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
        if not args.allow_extreme:
            # Don't update gcorr if correction is extreme
            v[v < 0.05] /= gcorr_corr["gcorr"][k][v < 0.05]
            v[v > 5] /= gcorr_corr["gcorr"][k][v > 5]
            for v0, val in enumerate(v):
                if val > 1.2:
                    if v0 != 0:
                        v[v0] = v[v0 - 1]
                    else:
                        v[v0] = 1.2
                if val < 0.2:
                    if v0 != 0:
                        v[v0] = v[v0 - 1]
                    else:
                        v[v0] = 0.2

        ax_tot[i].plot(v)
        ax_tot[i].set_title("{} total gcorr".format(k))
        ax_corr[i].plot(gcorr_corr["gcorr"][k])
        ax_corr[i].set_title("{} gcorr corr".format(k))

    print(gcorr["gcorr"])
    fig_tot.savefig(
        os.path.join(
            rundir,
            "plots",
            "gcorr_tot_{}_iter{}.png".format(tag, iternum),
        )
    )
    fig_corr.savefig(
        os.path.join(
            rundir,
            "plots",
            "gcorr_corr_{}_iter{}.png".format(tag, iternum),
        )
    )

    np.savez_compressed(ref_file, **gcorr)

    if iternum > 0:
        if args.keep_iters:
            # keep outputs from previous iteration
            rundirf_iter = os.path.join(rundir, tag, "iter{:03d}".format(iternum - 1))
            os.mkdir(rundirf_iter)
            sp.call(
                "rsync -a {}/bandpowers* {}/.".format(rundirf, rundirf_iter).split()
            )
            sp.call("rsync -a {}/transfer* {}/.".format(rundirf, rundirf_iter).split())
            sp.call("rsync -a {}/ERROR* {}/.".format(rundirf, rundirf_iter).split())
            sp.call("rsync -a {}/logs* {}/.".format(rundirf, rundirf_iter).split())
            sp.call("rsync -a {}/gcorr* {}/.".format(rundirf, rundirf_iter))

        # Remove transfer functions and bandpowers from run directory
        sp.call("rm -rf {}/bandpowers*".format(rundirf).split())
        sp.call("rm -rf {}/transfer*".format(rundirf).split())
        sp.call("rm -rf {}/ERROR*".format(rundirf).split())
        sp.call("rm -rf {}/logs".format(rundirf).split())

# Submit a first job that reloads gcorr and computes the transfer function
# that will be used by all the other seeds
print("Submitting first job")
gt.run_xfaster_gcorr(checkpoint="transfer", reload_gcorr=True, **run_opts)

transfer_exists = {}
for tag in tags:
    transfer_exists[tag] = False
while not np.all(list(transfer_exists.values())):
    # wait until transfer functions are done to submit rest of jobs
    for tag in tags:
        rundirf = os.path.join(rundir, tag)
        transfer_files = glob.glob(
            os.path.join(rundirf, "transfer_all*{}.npz".format(tag))
        )
        transfer_exists[tag] = bool(len(transfer_files))

    print("transfer exists: ", transfer_exists)
    if not args.submit:
        if np.all(list(transfer_exists.values())):
            break
        raise RuntimeError(
            "Some/all transfer functions not made: {}".format(transfer_exists)
        )
    else:
        time.sleep(15)

# Once transfer function is done, all other seeds can run
print("Submitting jobs for all seeds")
gt.run_xfaster_gcorr(
    checkpoint="bandpowers",
    sim_index=1,
    num_sims=int(g_cfg["gcorr_opts"]["nsim"]) - 1,
    **run_opts,
)

# If running jobs, wait for them to complete before computing gcorr factors
if args.submit:
    print("Waiting for jobs to complete...")
    check_cmd = "squeue -u $USER | grep xfast"
    while sp.call(check_cmd, shell=True, stdout=sp.DEVNULL) == 0:
        jobs = int(sp.check_output("{} | wc -l".format(check_cmd), shell=True))
        print("Number of jobs left:", jobs)
        time.sleep(10)

print("Computing new gcorr factors")
for tag in tags:
    out = gt.compute_gcal(g_cfg, output=run_name_iter, output_tag=tag)
    print("New gcorr correction computed (should converge to 1): ", out["gcorr"])

for tag in tags:
    bp_files = glob.glob("{}/{}/bandpowers*".format(rundir, tag))
    error_files = glob.glob("{}/{}/ERROR*".format(rundir, tag))
    print("{} completed: {}".format(tag, len(bp_files)))
    print("{} error: {}".format(tag, len(error_files)))
