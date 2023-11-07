import os
import glob
import copy
import time
import numpy as np
from . import parse_tools as pt
import subprocess as sp


def get_gcorr_config(filename):
    """
    Return a dictionary of options for running the gcorr iteration scripts.

    Arguments
    ---------
    filename : str or dict
        Filename to load.

    Returns
    -------
    cfg : dict
    """
    if isinstance(filename, dict):
        return filename

    assert os.path.exists(filename), "Missing config file {}".format(filename)

    from configparser import ConfigParser
    from ast import literal_eval

    cfg = ConfigParser()
    cfg.read(filename)

    out = {}
    for sec, sec_items in cfg.items():
        if not len(sec_items):
            continue
        out[sec] = {}
        for k, v in sec_items.items():
            if k in ["map_tags"]:
                v = [x.strip() for x in v.split(",")]
            else:
                try:
                    v = literal_eval(v)
                except:
                    pass
            if isinstance(v, str) and v.lower() in cfg.BOOLEAN_STATES:
                v = cfg.getboolean(sec, k)

            out[sec][k] = v

    return out


def get_next_iter(output_root="xfaster_gcal", output_tag=None):
    """ """
    if not os.path.exists(output_root):
        return 0

    tag = "" if output_tag is None else "_{}".format(output_tag)
    pattern = os.path.join(output_root, "*", "gcorr_total{}_iter*.npz".format(tag))
    return len(glob.glob(pattern))


def xfaster_gcorr(
    output_root="xfaster_gcal",
    output_tag=None,
    data_subset="full",
    force_restart=False,
    null=False,
    sim_index=0,
    num_sims=1,
    num_jobs=1,
    submit=False,
    **opts,
):
    """
    Run XFaster for the gcorr calculation.

    Arguments
    ---------
    output_root : str
        Output root where the data product will be stored.
    output_tag : str
        Map tag to analyze
    data_subset : str
        Data subset directory from which to load data maps.  The glob pattern
        for the `data_subset` XFaster option is built from this string for each
        map tag as `<data_subset>/*_<tag>`.
    null : bool
        If True, this is a null test run.
    force_restart : bool
        If True, restart iterations from 0.
    sim_index : int
        First sim index to run
    num_sims : int
        Number of sims to run
    num_jobs : int
        Number of jobs to split each sim ensemble into
    submit : bool
        If True, submit jobs to a cluster.
        Requires submit_opts section to be present in the config.
    opts :
        Remaining options are passed directly to `xfaster_run` or
        `xfaster_submit`.
    """
    # If output root doesn't exist or force_restart, we start from scratch
    if force_restart:
        tag = "" if not output_tag else "_{}".format(output_tag)
        gfiles = glob.glob(os.path.join(output_root, "*", "gcorr*{}*.npz".format(tag)))
        for f in gfiles:
            os.remove(f)
    iternum = get_next_iter(output_root, output_tag)
    print("Starting {} iteration {}".format(output_tag, iternum))

    if null:
        assert opts["noise_type"] is not None, "Missing noise_type"
        opts["sim_data_components"] = ["signal", "noise"]
    else:
        opts["noise_type"] = None
        opts["sim_data_components"] = ["signal"]

    opts["output_root"] = output_root
    opts["output_tag"] = output_tag
    opts["apply_gcorr"] = iternum > 0
    opts["reload_gcorr"] = iternum > 0
    opts["sim_data"] = True
    opts["save_sim_data"] = True
    opts["qb_only"] = True
    opts["sim_index_default"] = sim_index
    opts["num_sims"] = num_sims

    if output_tag is None:
        tag = ""
        groot = output_root
    else:
        tag = "_{}".format(output_tag)
        groot = os.path.join(output_root, output_tag)

    opts["data_subset"] = os.path.join(data_subset, "*{}".format(tag))
    gfile = os.path.join(groot, "gcorr_total{}.npz".format(tag))
    opts["gcorr_file"] = gfile
    if iternum > 0:
        assert os.path.exists(gfile), "Missing gcorr file {}".format(gfile)

    from .xfaster_exec import xfaster_submit, xfaster_run

    if submit:
        jobs = []

        # submit first job to compute transfer function
        opts["sim_index_default"] = sim_index
        opts["num_sims"] = 1 if num_jobs > 1 else num_sims
        jobs.extend(xfaster_submit(**opts))

        # submit remaining jobs to compute bandpowers for the rest of the
        # ensemble
        if num_sims > 1 and num_jobs > 1:
            opts["checkpoint"] = "bandpowers"
            opts["dep_afterok"] = [jobs[0]]

            idxs = np.array_split(np.arange(sim_index, sim_index + num_sims), num_jobs)
            for idx in idxs:
                opts["sim_index_default"] = idx[0]
                opts["num_sims"] = len(idx)
                jobs.extend(xfaster_submit(**opts))

        return jobs
    else:
        try:
            xfaster_run(**opts)
        except RuntimeError:
            pass


def compute_gcal(
    output_root="xfaster_gcal", output_tag=None, null=False, num_sims=1, fit_hist=False
):
    """
    Compute gcorr calibration

    Arguments
    ---------
    output_root : str
        Output root where the data product will be stored.
    output_tag : str
        Map tag to analyze
    null : bool
        If True, this is a null test dataset.
    num_sims : int
        Number of sim bandpowers to expect in the ensemble.
    fit_hist : bool
        If True, fit the bandpower histogram to a lognorm distribution to
        compute the calibration factor.  Otherwise, uses the simple variance of
        the distribution.
    """
    if null:
        # no sample variance used for null tests
        fish_name = "invfish_nosampvar"
    else:
        fish_name = "inv_fish"

    # use gauss model for null bandpowers
    def gauss(qb, amp, width, offset):
        # width = 0.5*1/sig**2
        # offset = mean
        return amp * np.exp(-width * (qb - offset) ** 2)

    # use lognormal model for signal bandpowers
    def lognorm(qb, amp, width, offset):
        return gauss(np.log(qb), amp, width, offset)

    # Compute the correction factor
    if output_tag is not None:
        output_root = os.path.join(output_root, output_tag)
        output_tag = "_{}".format(output_tag)
    else:
        output_tag = ""

    file_glob = os.path.join(output_root, "bandpowers_sim*{}.npz".format(output_tag))
    files = sorted(glob.glob(file_glob))
    efiles = glob.glob(
        os.path.join(output_root, "ERROR_" + os.path.basename(file_glob))
    )
    nf = len(files) + len(efiles)
    if nf != num_sims:
        raise OSError(
            "Found {} bandpowers files in {}, expected {}".format(
                nf, output_root, num_sims
            )
        )

    out = {"data_version": 1}
    inv_fishes = None
    qbs = {}

    for filename in files:
        bp = pt.load_and_parse(filename)
        inv_fish = np.diag(bp[fish_name])
        check = pt.arr_to_dict(inv_fish < 0, bp["qb"])
        bad = False
        for k, v in check.items():
            if not k.startswith("cmb_"):
                continue
            spec = k.split("_")[1]
            # ignore negative fisher values in off-diagonals
            if spec in ["te", "eb", "tb"]:
                continue
            # ignore negative fisher values in first bin
            if np.any(v[1:]):
                bad = True
                break
        if bad:
            # this happens rarely and we won't use those sims
            check = np.where(pt.dict_to_arr(check, flatten=True))[0]
            print("Found negative fisher values in {}: {}".format(filename, check))
            continue

        if inv_fishes is None:
            inv_fishes = inv_fish
        else:
            inv_fishes = np.vstack([inv_fishes, inv_fish])

        for stag, qb1 in bp["qb"].items():
            if not stag.startswith("cmb_"):
                continue
            spec = stag.split("_")[1]
            if spec not in qbs:
                qbs[spec] = qb1
            else:
                qbs[spec] = np.vstack([qbs[spec], qb1])

    # Get average XF-estimated variance
    xf_var_mean = np.mean(inv_fishes, axis=0)
    xf_var = pt.arr_to_dict(xf_var_mean, bp["qb"])

    out["bin_def"] = bp["bin_def"]
    out["gcorr"] = {}

    import scipy.optimize as opt

    for spec in qbs:
        stag = "cmb_{}".format(spec)
        nbins = len(out["bin_def"][stag])
        out["gcorr"][spec] = np.ones(nbins)

        if not fit_hist:
            fit_variance = np.var(qbs[spec], axis=0)
            out["gcorr"][spec] = xf_var[stag] / fit_variance
            continue

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
            except RuntimeError as e:
                print(
                    "Error computing gcorr for {} bin {}: {}".format(spec, b0, str(e))
                )
                out["gcorr"][spec][b0] = np.nan

    outfile = os.path.join(output_root, "gcorr_corr{}.npz".format(output_tag))
    pt.save(outfile, **out)
    return out


def apply_gcal(
    output_root="xfaster_gcal", output_tag=None, iternum=0, allow_extreme=False
):
    """
    Apply gcorr correction to the previous iteration's gcorr file.

    Arguments
    ---------
    output_root : str
        Output root where the data product will be stored.
    output_tag : str
        Map tag to analyze
    iternum : int
        Iteration number
    allow_extreme : bool
        Do not clip gcorr corrections that are too large at each iteration.  Try
        this if iterations are not converging.
    """
    plotdir = os.path.join(output_root, "plots")
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    gdir = os.path.join(output_root, "gcorr")
    if not os.path.exists(gdir):
        os.mkdir(gdir)

    if output_tag is not None:
        output_root = os.path.join(output_root, output_tag)
        output_tag = "_{}".format(output_tag)
    else:
        output_tag = ""

    gfile = os.path.join(output_root, "gcorr_total{}.npz".format(output_tag))
    cfile = os.path.join(output_root, "gcorr_corr{}.npz".format(output_tag))
    gcorr_corr = pt.load_and_parse(cfile)

    if iternum > 0:
        gcorr = pt.load_and_parse(gfile)
        gcorr["gcorr"] = pt.arr_to_dict(
            pt.dict_to_arr(gcorr["gcorr"], flatten=True)
            * pt.dict_to_arr(gcorr_corr["gcorr"], flatten=True),
            gcorr["gcorr"],
        )
    else:
        gcorr = copy.deepcopy(gcorr_corr)

    pt.save(gfile, **gcorr)

    import matplotlib.pyplot as plt

    fig_tot, ax_tot = plt.subplots(2, 3, sharex=True, sharey="row")
    fig_corr, ax_corr = plt.subplots(2, 3, sharex=True, sharey="row")
    ax_tot = ax_tot.flatten()
    ax_corr = ax_corr.flatten()
    for i, (k, v) in enumerate(gcorr["gcorr"].items()):
        v[0] = 0.5
        if k in ["te", "eb", "tb"]:
            # We don't compute gcorr for off-diagonals
            v[:] = 1
        if not allow_extreme:
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

    ftot = os.path.join(
        plotdir, "gcorr_total{}_iter{:03d}.png".format(output_tag, iternum)
    )
    fig_tot.tight_layout()
    fig_tot.savefig(ftot, bbox_inches="tight")
    fcorr = ftot.replace("gcorr_total", "gcorr_corr")
    fig_corr.tight_layout()
    fig_corr.savefig(fcorr, bbox_inches="tight")

    # Save a copy of the gcorr iteration files
    pt.save(ftot.replace(".png", ".npz").replace(plotdir, gdir), **gcorr)
    pt.save(fcorr.replace(".png", ".npz").replace(plotdir, gdir), **gcorr_corr)

    return gcorr


def process_gcorr(
    output_root="xfaster_gcal",
    output_tag=None,
    null=False,
    num_sims=1,
    gcorr_fit_hist=False,
    allow_extreme=False,
    keep_iters=False,
):
    """
    Run gcorr analysis on a complete ensemble of sim bandpowers. This function
    runs ``compute_gcal`` to calculate a correction-to-gcorr, stored in the
    running directory as ``gcorr_corr_<tag>.npz``.  It then runs ``apply_gcal``
    to increment the running total gcorr, stored in ``gcorr_total_<tag>,npz``.
    Copies of the correction and total files for the current iteration are
    stored in the ``<output_root>/gcorr`` directory, and plots of these two
    outputs are stored in the ``<output_root>/plots`` directory.  The current
    iteration is determined by counting the number of ``gcorr_total`` files.

    Arguments
    ---------
    Arguments
    ---------
    output_root : str
        Output root where the data product will be stored.
    output_tag : str
        Map tag to analyze
    null : bool
        If True, this is a null test dataset.
    num_sims : int
        Number of sim bandpowers to expect in the ensemble.
    gcorr_fit_hist : bool
        If True, fit the bandpower histogram to a lognorm distribution to
        compute the calibration factor.  Otherwise, uses the simple variance of
        the distribution.
    allow_extreme : bool
        Do not clip gcorr corrections that are too large at each iteration.  Try
        this if iterations are not converging.
    keep_iters : bool
        If True, store the transfer function and bandpowers outputs from each
        iteration in a separate sub-directory, rather than deleting these files.
    """
    iternum = get_next_iter(output_root, output_tag)

    # Compute gcorr correction from bandpower distribution
    out = compute_gcal(
        output_root=output_root,
        output_tag=output_tag,
        num_sims=num_sims,
        null=null,
        fit_hist=gcorr_fit_hist,
    )

    # Apply correction to cumulative gcorr
    gcorr = apply_gcal(
        output_root=output_root,
        output_tag=output_tag,
        iternum=iternum,
        allow_extreme=allow_extreme,
    )

    # Cleanup output directories
    if output_tag is not None:
        output_root = os.path.join(output_root, output_tag)
        ftag = "_{}".format(output_tag)
    else:
        ftag = ""

    bp_files = glob.glob(os.path.join(output_root, "bandpowers_sim*{}.npz"))
    error_files = glob.glob(os.path.join(output_root, "ERROR_bandpowers_sim*{}.npz"))
    print("{} iter {} completed: {}".format(output_tag, iternum, len(bp_files)))
    print("{} iter {} error: {}".format(output_tag, iternum, len(error_files)))

    flist = ["bandpowers_sim", "transfer"]
    if len(error_files) > 0:
        flist += ["ERROR_bandpowers_sim"]
    flist = ["{}/{}*{}.npz".format(output_root, f, ftag) for f in flist]
    flist += ["{}/logs*".format(output_root)]

    if keep_iters:
        # Keep iteration output files
        iterdir = os.path.join(output_root, "iter{:03d}".format(iternum))
        os.mkdir(iterdir)
        for f in flist:
            sp.check_call("rsync -a {} {}/".format(f, iterdir), shell=True)

    # Remove transfer functions and bandpowers from run directory
    for f in flist:
        sp.check_call("rm -rf {}".format(f), shell=True)

    print("{} iter {} gcorr correction: {}".format(output_tag, iternum, out["gcorr"]))
    print("{} iter {} total gcorr: {}".format(output_tag, iternum, gcorr["gcorr"]))
