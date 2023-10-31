import os
import glob
from configparser import ConfigParser
from . import parse_tools as pt
import numpy as np


def get_gcorr_config(filename):
    """
    Return a ConfigParser for running the gcorr iteration scripts.

    Arguments
    ---------
    filename : str or ConfigParser
        Filename to load.

    Returns
    -------
    cfg : ConfigParser
    """
    if isinstance(filename, ConfigParser):
        return filename

    assert os.path.exists(filename), "Missing config file {}".format(filename)

    cfg = ConfigParser()
    cfg.read(filename)

    return cfg


def run_xfaster_gcorr(
    cfg,
    output="xfaster_gcal",
    apply_gcorr=True,
    reload_gcorr=False,
    checkpoint=None,
    sim_index=0,
    num_sims=1,
    submit=False,
    omp_threads=None,
):
    """
    Run XFaster for the gcorr calculation.

    Arguments
    ---------
    cfg : str or ConfigParser
        Configuration to use
    output : str
        Output root where the data product will be stored.
    apply_gcorr : bool
        Apply a g-correction in the XFaster computation
    reload_gcorr : bool
        Reload the gcorr factor
    checkpoint : str
        XFaster checkpoint
    sim_index : int
        First sim index to run
    num_sims : int
        Number of sims to run
    submit : bool
        If True, submit jobs to a cluster.
        Requires submit_opts section to be present in the config
    omp_threads : int
        Override omp_threads option from config file.
    """
    cfg = get_gcorr_config(cfg)

    # Change XFaster options here to suit your purposes
    opts = dict(
        likelihood=False,
        residual_fit=False,
        foregorund_fit=False,
        # change options below for your purposes
        tbeb=True,
        bin_width=25,
        lmin=2,
        lmax=500,
    )

    xopts = cfg["xfaster_opts"]
    gopts = cfg["gcorr_opts"]
    sopts = cfg["submit_opts"]

    opts.update(**cfg["xfaster_opts"])

    null = cfg.getboolean("gcorr_opts", "null")
    tags = gopts["map_tags"].split(",")

    if null:
        opts["noise_type"] = xopts["noise_type"]
        opts["sim_data_components"] = ["signal", "noise"]
    else:
        opts["noise_type"] = None
        opts["sim_data_components"] = ["signal"]

    opts["output_root"] = os.path.join(gopts["output_root"], output)
    opts["apply_gcorr"] = apply_gcorr
    opts["reload_gcorr"] = reload_gcorr
    opts["checkpoint"] = checkpoint
    opts["sim_data"] = True

    if submit:
        opts.update(**sopts)
        if omp_threads is not None:
            opts["omp_threads"] = omp_threads

    seeds = list(range(sim_index, sim_index + num_sims))

    from .xfaster_exec import xfaster_submit, xfaster_run

    for tag in tags:
        opts["output_tag"] = tag
        opts["data_subset"] = os.path.join(gopts["data_subset"], "*{}".format(tag))
        opts["gcorr_file"] = os.path.abspath(
            os.path.join(opts["output_root"], tag, "gcorr_{}_total.npz".format(tag))
        )

        for s in seeds:
            opts["sim_index_default"] = s
            if submit:
                xfaster_submit(**opts)
            else:
                xfaster_run(**opts)


def compute_gcal(cfg, output="xfaster_gcal", output_tag=None, fit_hist=False):
    """
    Compute gcorr calibration

    Arguments
    ---------
    cfg : str or ConfigParser
        Configuration to use
    output : str
        Output root where the data product will be stored.
    output_tag : str
        Map tag to analyze
    fit_hist : bool
        If True, fit the bandpower distribution to a histogram to compute the
        calibration factor.  Otherwise, uses the simple variance of the
        distribution.
    """
    cfg = get_gcorr_config(cfg)

    null = cfg.getboolean("gcorr_opts", "null")
    if null:
        # no sample variance used for null tests
        fish_name = "invfish_nosampvar"
    else:
        fish_name = "inv_fish"

    output_root = os.path.join(cfg["gcorr_opts"]["output_root"], output)

    specs = ["tt", "ee", "bb", "te", "eb", "tb"]
    nsim = cfg.getint("gcorr_opts", "nsim")

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
    if not len(files):
        raise OSError("No bandpowers files found in {}".format(output_root))

    out = {"data_version": 1}
    inv_fishes = None
    qbs = {}

    for spec in specs:
        qbs[spec] = None

    for filename in files:
        bp = pt.load_and_parse(filename)
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

    import scipy.optimize as opt

    for spec in specs:
        stag = "cmb_{}".format(spec)
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
            except RuntimeError:
                out["gcorr"][spec][b0] = np.nan

    outfile = os.path.join(output_root, "gcorr_corr{}.npz".format(output_tag))
    np.savez_compressed(outfile, **out)
    return out
