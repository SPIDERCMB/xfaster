import numpy as np
import matplotlib.pyplot as plt
import xfaster as xf
from xfaster import xfaster_tools as xft

# First, load up inputs to our sims so we can check how well they're recovered
# (bearing in mind, this is a single sim, so noise fluctuations and sample
# variance will cause scatter.
r_in = 1.0
Dls_in = xft.get_camb_cl(r=r_in, lmax=500, lfac=True)
Fl_in = np.loadtxt("maps_example/transfer_example.txt")

# load up bandpowers file, where most of the useful stuff is stored
bp = xf.load_and_parse("outputs_example/95x150/bandpowers_95x150.npz")
ee_bin_centers = bp["ellb"]["cmb_ee"]  # weighted bin centers
ee_specs = bp["cb"]["cmb_ee"]  # estimated CMB spectra with ell*(ell+1)/(2pi) factors
ee_errs = bp["dcb"]["cmb_ee"]  # estimated CMB error bars
spec_cov = bp["cov"]  # Nspec * Nbin square covariance matrix
ee_transfer_150 = bp["qb_transfer"]["cmb_ee"]["150"]  # transfer function using the same bins

fig, axs = plt.subplots(3, 1, figsize=(4,6))
axs[0].plot(Fl_in[:500], color="k", label="Input Transfer Function")
axs[0].plot(ee_bin_centers, ee_transfer_150, label="Estimated Transfer Function")
axs[0].set_ylabel(r"$F_\ell^{EE}$")
axs[0].set_xlabel(r"$\ell$")
axs[0].legend()

axs[1].plot(Dls_in[1], color="k", label="Input CMB")
axs[1].errorbar(ee_bin_centers, ee_specs, ee_errs, label="Output CMB Estimate")
axs[1].set_ylabel(r"$\ell(\ell+1)C_\ell^{EE}/2\pi\, [\mu K_{CMB}]$")
axs[1].set_xlabel(r"$\ell$")
axs[1].legend()

# Now get r-likelihood-- should be near the input r=1, but with scatter since it's
# just one sim realization
lk = xf.load_and_parse("outputs_example/95x150/like_mcmc_95x150.npz")

axs[2].axvline(r_in, color="k", label="Input r")
axs[2].hist(lk["samples"], label="r posterior")
axs[2].set_xlabel(r"$r$")
axs[2].legend()
plt.tight_layout()
plt.savefig("outputs_example.png")
plt.show()
