import os
import numpy as np
import healpy as hp
import xfaster as xf

"""
This scipt will generate the ensemble of maps needed to run the
xfaster_example.py script, including fake data maps, signal sims,
noise sims, and masks.

Script should take <10 minutes and write about 350 M of maps to disk
"""

# set up options
nsim = 100
nside = 256
do_fg = True

betad = 1.54
ref_freq = 359.7

tags = [95, 150]
freqs = [94.7, 151.0]
scales = [xf.spec_tools.scale_dust(f, nu0=ref_freq, beta=betad) for f in freqs]
fwhms = [41.0, 29.0]  # arcmin
gnoise = [4.5, 3.5]  # Temp std, uK_CMB
data_noise_scale = 0.85  # make data noise less than sims by 15%

mask_path = "maps_example/masks_rectangle"
data_path = "maps_example/data_raw/full"
data_fg_path = "maps_example/data_cmbfg/full"
sig_path = "maps_example/signal_synfast/full"
noise_path = "maps_example/noise_gaussian/full"
fg_path = "maps_example/foreground_gaussian/full"

for p0 in [mask_path, data_path, data_fg_path, sig_path, noise_path, fg_path]:
    if not os.path.exists(p0):
        os.makedirs(p0)

# code expects CAMB default outputs (with ell*(ell+1)/(2pi) factor)
# with ell vector, and only ell>=2
ell = np.arange(2001)
lfac = ell * (ell + 1) / (2 * np.pi)

# spectrum for signal sims-- synfast expects Cls
spec_file = os.path.join(*sig_path.split("/")[:-1], "spec_signal_synfast.dat")
if os.path.exists(spec_file):
    cls = xf.spec_tools.load_camb_cl(spec_file, lfac=False)
else:
    cls = xf.get_camb_cl(r=1.0, lmax=2000, lfac=False)
    # write to disk for transfer function
    dls = np.vstack([ell[2:], (lfac * cls)[:, 2:]])
    np.savetxt(spec_file, dls.T)

# spectrum for foreground sims
fg_spec_file = os.path.join(*fg_path.split("/")[:-1], "spec_foreground_gaussian.dat")
if os.path.exists(spec_file):
    cls_fg = xf.spec_tools.load_camb_cl(spec_file, lfac=False)
else:
    cls_fg = xf.spec_tools.dust_model(ell, lfac=False)
    # write to disk for transfer function
    dls_fg = np.vstack([ell[2:], (lfac * cls_fg)[:, 2:]])
    np.savetxt(fg_spec_file, dls_fg.T)

# load a filter transfer function to smooth by
fl = np.loadtxt("maps_example/transfer_example.txt")
mask = None


def read_map(filename, fill=None):
    fields = 0 if "mask" in filename else (0, 1, 2)
    data = np.asarray(hp.read_map(filename, field=fields))
    if fill is not None:
        data[hp.mask_bad(data)] = fill
    return data


def write_map(filename, data, mask=None):
    if mask is not None:
        data[..., ~mask] = hp.UNSEEN
    hp.write_map(filename, data, partial=True, overwrite=True)


mask = None
mask_write = None

# make celestial rectangular coordinate mask
for tag in tags:
    mask_file = os.path.join(mask_path, "mask_map_{}.fits".format(tag))
    if os.path.exists(mask_file):
        mask = read_map(mask_file, fill=0).astype(bool)
        break

    if mask is None:
        latrange = [-55, -15]
        lonrange = [18, 80]
        theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        lat = 90 - np.rad2deg(theta)
        lon = np.rad2deg(phi)
        lon[lon > 180] -= 360
        mask = np.logical_and(
            np.logical_and(lon > lonrange[0], lon < lonrange[1]),
            np.logical_and(lat > latrange[0], lat < latrange[1]),
        )
        mask_write = mask.copy().astype(float)
        mask_write[~mask] = hp.UNSEEN  # to reduce size of file on disk
        # Because using polarization, need I/Q/U mask
        mask_write = np.tile(mask_write, [3, 1])

    write_map(mask_file, mask_write)
    print("Wrote mask map {}".format(tag))

sig_cache = {}
fg_cache = {}


def sim_signal(isim, itag):
    if isim not in sig_cache:
        np.random.seed(isim)
        sig = hp.synfast(cls, nside=nside, new=True, pixwin=True)
        sig = hp.smoothing(sig, beam_window=np.sqrt(fl))
        sig_cache[isim] = np.asarray(sig)
    return hp.smoothing(sig_cache[isim], np.deg2rad(fwhms[itag] / 60.0), verbose=False)


def sim_noise(isim, itag):
    np.random.seed(nsim * (1 + itag) + isim)
    noise = np.tile(np.zeros(mask.size, dtype=float), (3, 1))
    noise[0][mask] = np.random.normal(scale=gnoise[itag], size=np.sum(mask))
    noise[1][mask] = np.random.normal(
        scale=gnoise[itag] * np.sqrt(2), size=np.sum(mask)  # pol
    )
    noise[2][mask] = np.random.normal(
        scale=gnoise[itag] * np.sqrt(2), size=np.sum(mask)  # pol
    )
    return hp.smoothing(noise, beam_window=np.sqrt(fl))


def sim_fg(isim, itag):
    if isim not in fg_cache:
        np.random.seed(nsim * (1 + len(tags)) + isim)
        fg = hp.synfast(cls_fg, nside=nside, new=True, pixwin=True)
        fg = hp.smoothing(fg, beam_window=np.sqrt(fl))
        fg_cache[isim] = np.asarray(fg)
    return scales[itag] * hp.smoothing(
        fg_cache[isim], np.deg2rad(fwhms[itag] / 60.0), verbose=False
    )


# make data, signal, noise, fg sims
for i in range(nsim + 1):
    for ti, tag in enumerate(tags):
        if i == 0:
            data_file = os.path.join(data_path, "map_{}.fits".format(tag))
            data_fg_file = os.path.join(data_fg_path, "map_{}.fits".format(tag))
            data = None
            if not os.path.exists(data_file):
                data = sim_signal(i, ti) + data_noise_scale * sim_noise(i, ti)
                write_map(data_file, data, mask)
                print("Wrote data signal+noise map {}".format(tag))
            if do_fg and not os.path.exists(data_fg_file):
                if data is None:
                    data = read_map(data_file)
                write_map(data_fg_file, data + sim_fg(i, ti), mask)
                print("Wrote data signal+noise+fg map {}".format(tag))
        else:
            sig_file = os.path.join(sig_path, "map_{}_{:04}.fits".format(tag, i - 1))
            noise_file = os.path.join(
                noise_path, "map_{}_{:04}.fits".format(tag, i - 1)
            )
            fg_file = os.path.join(fg_path, "map_{}_{:04}.fits".format(tag, i - 1))

            comps = []
            if not os.path.exists(sig_file):
                write_map(sig_file, sim_signal(i, ti), mask)
                comps += ["sig"]
            if not os.path.exists(noise_file):
                write_map(noise_file, sim_noise(i, ti), mask)
                comps += ["noise"]
            if do_fg and not os.path.exists(fg_file):
                write_map(fg_file, sim_fg(i, ti), mask)
                comps += ["fg"]
            if len(comps):
                print(
                    "Wrote {} map {} {} / {}".format(", ".join(comps), tag, i - 1, nsim)
                )

    sig_cache.pop(i, None)
    fg_cache.pop(i, None)
