from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import warnings

__all__ = [
    "wigner3j",
    "get_camb_cl",
    "load_camb_cl",
    "scale_dust",
    "dust_model",
]

k = 1.38064852e-23  # Boltzmann constant
h = 6.626070040e-34  # Planck constant
hk = h / k * 1.0e9  # conversion from GHz / K to unitless value
T_dust = 19.6  # dust temperature in K
T_cmb = 2.72548  # CMB blackbody temperature in K


def blackbody(nu, nu0=None, T=T_cmb):
    """
    The blackbody function at temperature ``T`` and frequency ``nu``, optionally
    relative to the value at reference frequency ``nu0``.

    Arguments
    ---------
    nu : float
        Frequency in GHz.
    nu0 : float
        Reference frequency in GHz.  If not supplied, return unreferenced
        blackbody.
    T : float
        Blackbody temperature.

    Returns
    -------
    blackbody : float
        B(nu, T) / B(nu0, T)
    """
    x = hk * nu / T
    if nu0 is None:
        return x ** 3 / (np.exp(x) - 1)
    x0 = hk * nu0 / T
    return (x / x0) ** 3 * (np.exp(x0) - 1) / (np.exp(x) - 1)


def rj2bb(nu, nu0=None, T=T_cmb):
    """
    Conversion from Rayleigh-Jeans units to Blackbody temperature units

    Arguments
    ---------
    nu : float
        Frequency in GHz.
    nu0 : float
        Reference frequency in GHz.  Ignored if not supplied.
    T : float
        Blakbody temperature

    Returns
    -------
    cal_fac : float
        Value by which to multiply a RJ temperature to get a CMB temperature.
        If ``nu0`` is given, return the value relative to that at the reference
        frequency.
    """
    x = hk * nu / T
    if nu0 is None:
        return ((np.exp(x) - 1.0) / x) ** 2 / np.exp(x)
    x0 = hk * nu0 / T
    return ((np.exp(x) - 1.0) / (np.exp(x0) - 1.0) * x0 / x) ** 2 * np.exp(x0 - x)


def scale_dust(nu1, nu2=None, nu0=359.7, beta=1.54, delta=False):
    """
    Get the factor by which you must multiply the cross spectrum from maps of
    frequencies ``nu1`` and ``nu2`` to match the dust power at ``nu0`` given
    spectral index ``beta``.  If ``nu2`` is not given, compute the map-domain
    factor for ``nu1`` alone.  Optionally include terms to construct the
    derivative with respect to ``beta`` if ``delta=True``.

    Arguments
    ---------
    nu1 : float
        Frequency of map0 in GHz.
    nu2 : float
        Frequency of map1 in GHz.  If not supplied, return the map-domain
        multiplicative factor for ``nu1`` only.
    nu0 : float
        Reference frequency from which to compute relative scaling in GHz.
    beta : float
        Dust spectral index.
    delta : bool
        If True, also return the multiplicative beta scaling and its first
        derivative.

    Returns
    -------
    freq_scale : float
        The relative scaling factor for the dust map or spectrum-- multiply by
        this number to get the dust map or spectrum at the reference frequency
    beta_scale : float
        The multiplicative scaling factor for beta-- multiply the
        frequency-scaled map or spectrum by ``beta_scale ** delta_beta`` to
        obtain the dust map or spectrum at the adjusted spectral index ``beta +
        delta_beta``.
    log_beta_scale : float
        The first derivative of the ``beta`` scaling-- multiply the
        frequency-scaled map or spectrum by this number to obtain the derivative
        of the frequency-scaled map or spectrum with respect to ``beta``.
    """
    bs = nu1 / nu0
    fs = (
        rj2bb(nu1, nu0=nu0, T=T_cmb)
        * blackbody(nu1, nu0=nu0, T=T_dust)
        * bs ** (beta - 2.0)
    )

    if nu2 is None:
        if delta:
            return (fs, bs, np.log(bs))
        return fs

    if delta:
        fs2, bs2, log_bs2 = scale_dust(nu2, nu0=nu0, beta=beta, delta=True)
        return (fs * fs2, bs * bs2, np.log(bs * bs2))

    return fs * scale_dust(nu2, nu0=nu0, beta=beta, delta=False)


def dust_model(ell, pivot=80, amp=34.0, index=-2.28, lfac=True):
    """
    Construct a power-law dust power spectrum.  Default parameter values are for
    the Planck LIV best-fit EE dust spectrum.

    The functional form is::

        model = amp * (ell / pivot) ** index

    Arguments
    ---------
    ell : array_like
        Multipoles over which to compute the model
    pivot : scalar
        Pivot ell at which to refence the amplitude
    amp : scalar
        Model amplitude
    index : scalar
        Model spectral index
    lfac: bool
        If True, multiply Cls by ell*(ell+1)/2/pi

    Returns
    -------
    model : array_like
        The dust model computed for each input ell value.
    """
    ell = np.asarray(ell)
    model = np.zeros(len(ell), dtype=float)
    model[ell > 1] = amp * (ell[ell > 1] / float(pivot)) ** (index + 2.0)
    if not lfac:
        ellfac = ell * (ell + 1) / 2.0 / np.pi
        model[ell > 1] /= ellfac[ell > 1]
    return model


def wigner3j(l2, m2, l3, m3):
    r"""
    Wigner 3j symbols computed for all valid values of ``L``, as in:

    .. math::

        \begin{pmatrix}
         \ell_2 & \ell_3 & L \\
         m_2 & m_3 & 0 \\
        \end{pmatrix}

    Arguments
    ---------
    l2, m2, l3, m3 : int
        The ell and m values for which to compute the symbols.

    Returns
    -------
    fj : array_like
        Array of size ``l2 + l3 + 2``, indexed by ``L``
    lmin : int
        The minimum value of ``L`` for which ``fj`` is non-zero.
    lmax : int
        The maximum value of ``L`` for which ``fj`` is non-zero.
    """
    import camb

    try:
        from camb.mathutils import threej
    except ImportError:
        from camb.bispectrum import threej
    arr = threej(l2, l3, m2, m3)

    lmin = np.max([np.abs(l2 - l3), np.abs(m2 + m3)])
    lmax = l2 + l3
    fj = np.zeros(lmax + 2, dtype=arr.dtype)
    fj[lmin : lmax + 1] = arr
    return fj, lmin, lmax


def get_camb_cl(r, lmax, nt=None, spec="total", lfac=True):
    """
    Compute camb spectrum with tensors and lensing.

    Parameter values are from arXiv:1807.06209 Table 1 Plik best fit

    Arguments
    ---------
    r : float
        Tensor-to-scalar ratio
    lmax : int
        Maximum ell for which to compute spectra
    nt : scalar, optional
        Tensor spectral index.  If not supplied, assumes
        slow-roll consistency relation.
    spec : string, optional
        Spectrum component to return.  Can be 'total', 'unlensed_total',
        'unlensed_scalar', 'lensed_scalar', 'tensor', 'lens_potential'.
    lfac: bool, optional
        If True, multiply Cls by ell*(ell+1)/2/pi

    Returns
    -------
    cls : array_like
        Array of spectra of shape (nspec, lmax + 1).
        Diagonal ordering (TT, EE, BB, TE).
    """
    # Set up a new set of parameters for CAMB
    import camb

    pars = camb.CAMBparams()

    # This function sets up CosmoMC-like settings, with one massive neutrino and
    # helium set using BBN consistency
    pars.set_cosmology(
        H0=67.32,
        ombh2=0.022383,
        omch2=0.12011,
        mnu=0.06,
        omk=0,
        tau=0.0543,
    )

    ln1010As = 3.0448

    pars.InitPower.set_params(As=np.exp(ln1010As) / 1.0e10, ns=0.96605, r=r, nt=nt)
    if lmax < 2500:
        # This results in unacceptable bias. Use higher lmax, then cut it down
        lmax0 = 2500
    else:
        lmax0 = lmax
    pars.set_for_lmax(lmax0, lens_potential_accuracy=2)
    pars.WantTensors = True
    pars.do_lensing = True

    # calculate results for these parameters
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=not lfac)

    totCL = powers[spec][: lmax + 1, :4].T

    return totCL


def load_camb_cl(filename, lmax=None, pol=None, lfac=True):
    """
    Load a CAMB spectrum from a text file.  Expects a file with at
    least two columns, where the first column contains ell values, and
    the rest are spectrum components (TT, EE, BB, TE).  Old-style CAMB
    files, where the spectra are ordered as (TT, TE, EE, BB), are
    re-indexed into the new-style ordering.

    Arguments
    ---------
    filename : str
        Path to spectrum file on disk.
    lmax : int
        If supplied, return spectrum up to this ell.  Otherwise, all ell's are
        included in the output.
    pol : bool
        If True, include polarization spectra in the output, if present in the
        file.  If False, include only the TT spectrum.  Otherwise, include all
        available spectra.
    lfac : bool
        If True, multiply Cls by ell*(ell+1)/2/pi

    Returns
    -------
    cls : array_like
        Array of spectra of shape (nspec, lmax + 1).
        Diagonal ordering (TT, EE, BB, TE, EB, TB).
    """

    data = np.loadtxt(filename, unpack=True)
    ell, data = data[0].astype(int), data[1:]

    # limit to lmax
    if lmax is not None:
        if lmax > ell.max():
            raise ValueError(
                "Require at least lmax={} in CAMB file, found {}".format(
                    lmax, ell.max()
                )
            )
        idx = ell <= lmax
        ell = ell[idx]
        data = data[..., idx]

    # include monopole and dipole
    if ell.min() > 0:
        n = ell.min()
        ell = np.append(np.arange(n), ell)
        data = np.append(np.zeros((len(data), n)), data, axis=1)

    # check polarization
    if pol is None:
        pol = len(data) > 1

    # select columns
    if pol:
        if len(data) == 1:
            raise ValueError(
                "Missing polarization spectra in CAMB file {}".format(filename)
            )
        if np.any(data[1, : ell.max() - 2] < 0):
            warnings.warn("Old CAMB format file {}. Re-indexing.".format(filename))
            order = [0, 2, 3, 1] if len(data) == 4 else [0, 3, 5, 1, 4, 2]
            data = data[order, :]
    else:
        data = data[[0], :]

    # check scaling
    if not lfac:
        f = ell * (ell + 1) / 2.0 / np.pi
        f[ell == 0] = 1
        data /= f[None, :]

    return data
