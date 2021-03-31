import os
import numpy as np
import healpy as hp
from xfaster import xfaster_tools as xft

"""
This scipt will generate the ensemble of maps needed to run the 
xfaster_example.py script, including fake data maps, signal sims,
noise sims, and masks.

Script should take <10 minutes and write about 240 M of maps to disk
"""

# set up options
nsim = 100
nside = 256
seed0 = 0
np.random.seed(seed0)

tags = [95, 150]
fwhms = [41., 29.] # arcmin
gnoise = [4.5, 3.5] # Temp std, uK_CMB
data_noise_scale = 0.85 # make data noise less than sims by 15%

mask_path = "maps_example/masks_rectangle"
data_path = "maps_example/data_raw/full"
sig_path = "maps_example/signal_synfast/full"
noise_path = "maps_example/noise_gaussian/full"

for p0 in [mask_path, data_path, sig_path, noise_path]:
    if not os.path.exists(p0):
        os.makedirs(p0)

# spectrum for signal sims-- synfast expects Cls
cls = xft.get_camb_cl(r=1., lmax=2000, lfac=False)

# write to disk for transfer function
# code expects CAMB default outputs (with ell*(ell+1)/(2pi) factor)
# with ell vector, and only ell>=2
ell = np.arange(2001)
lfac = ell*(ell+1)/(2*np.pi)
dls = np.vstack([ell[2:], (lfac * cls)[:, 2:]])
np.savetxt(os.path.join(*sig_path.split("/")[:-1], "spec_signal_synfast.dat"),
           dls.T)

# make celestial rectangular coordinate mask
latrange = [-55, -15]
lonrange = [18, 80]
theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
lat = 90 - np.rad2deg(theta)
lon = np.rad2deg(phi)
lon[lon > 180] -= 360
mask = np.logical_and(
    np.logical_and(lon > lonrange[0], lon < lonrange[1]),
    np.logical_and(lat > latrange[0], lat < latrange[1]))
mask_write = mask.copy().astype(float)
mask_write[~mask] = hp.UNSEEN # to reduce size of file on disk
for tag in tags:
    # Because using polarization, need I/Q/U mask
    mask_write = np.tile(mask_write, [3,1])
    hp.write_map(os.path.join(mask_path, "mask_map_{}.fits".format(tag)),
                 mask_write, partial=True, overwrite=True)
    print("Wrote mask map {}".format(tag))

# make data, signal, noise sims
for i in range(nsim + 1):
    sig = hp.synfast(cls, nside=nside)
    noise = np.zeros_like(sig)
    for ti, tag in enumerate(tags):
        sig_smooth = hp.smoothing(sig, np.deg2rad(fwhms[ti]/60.), verbose=False)
        noise[0][mask] = np.random.normal(scale=gnoise[ti], size=np.sum(mask))
        noise[1][mask] = np.random.normal(scale=gnoise[ti]*np.sqrt(2), #pol
                                          size=np.sum(mask))
        noise[2][mask] = np.random.normal(scale=gnoise[ti]*np.sqrt(2), #pol
                                          size=np.sum(mask))
        if i == 0:
            # call this data
            dat = sig_smooth + data_noise_scale * noise
            dat[:, ~mask] = hp.UNSEEN
            hp.write_map(os.path.join(data_path, "map_{}.fits".format(tag)),
                         dat, partial=True, overwrite=True)
            print("Wrote data map {}".format(tag))
        else:
            # signal and noise
            dat = sig_smooth + data_noise_scale * noise
            sig_smooth[:, ~mask] = hp.UNSEEN
            noise[:, ~mask] = hp.UNSEEN
            hp.write_map(os.path.join(sig_path,
                                      "map_{}_{:04}.fits".format(tag, i-1)),
                         sig_smooth, partial=True, overwrite=True)
            hp.write_map(os.path.join(noise_path,
                                      "map_{}_{:04}.fits".format(tag, i-1)),
                         noise, partial=True, overwrite=True)
            print("Wrote signal and noise map {} {}".format(tag, i-1))
            
            
