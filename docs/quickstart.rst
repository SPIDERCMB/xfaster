Quick Start
===========

In this page we'll go over the basics of using the code and its outputs.

Dependencies
------------

* ``numpy``
* ``healpy``
* ``camb``

XFaster is compatible with Python versions 3.0 and higher.

Installation
------------
After cloning the repo, install xfaster::

    $ python setup.py install

Setting up your data
--------------------

Maps
....

The code requires a certain directory structure for your input maps:

.. code-block:: text

    <data_root>/
    ├── data_<data_type>
    │   ├── <data_subset1>    
    │   │   ├── map_<tag1>.fits
    │   │   ├── ...
    │   │   ├── map_<tagN>.fits
    │   ├── <data_subset2> (same filenames as <data_subset1>)
    │   ├── ....
    │   ├── <data_subsetM>
    ├── signal_<signal_type>
    │   ├── spec_signal_<signal_type>.dat
    │   ├── <data_subset1>    
    │   │   ├── map_<tag1>_0000.fits
    │   │   ├── ...
    │   │   ├── map_<tag1>_####.fits
    │   │   ├── ...    
    │   │   ├── map_<tagN>_0000.fits
    │   │   ├── ...
    │   │   ├── map_<tagN>_####.fits    
    │   ├── ....
    │   ├── <data_subsetM> (same filenames as <data_subset1>)
    ├── noise_<noise_type> (same filenames as signal_<signal_type>)
    ├── masks_<mask_type>
    │   ├── mask_map_<tag1>.fits
    │   ├── ...
    │   ├── mask_map_<tagN>.fits		
    [[optional:]]
    ├── foreground_<foreground_type> (same filenames as signal_<signal_type>
    ├── templates_<template_type>
    │   ├── halfmission-1 (same filenames as data_<data_type>)
    │   ├── halfmission-2 (same filenames as data_<data_type>)
    └── reobs_planck (same filenames as templates_<template_type>, used if sub_planck=True for null tests)

The required types of maps are data, signal simulations, noise simulations, and masks.
Each map should have a tag, which is used consistently across these different map types.
This would typically be a frequency or some subset of detectors.
For example, SPIDER has two tags: ``95`` and ``150``.

Further subsetting of the data, ie., by time splits, is done through use of the ``<data_subset>>`` directories.
These should contain independent subsets of data, as they are used to construct cross-spectra.
For example, SPIDER has four subsets, labeled ``1of4``, ``2of4``, etc, to contain maps from every fourth set of interleaved ten-minute chunks.

For each map tag and for each data subset in your data directory, you must have at least one (but generally at least 100, for reasonable statistics) maps in your signal and noise directories.
These have the same directory structures, but the maps now have additional numerical tags for each realization of randomly generated signal or noise.
The tags don't have to start at 0 or be contiguous-- the code will simply look for any maps in your sim directories that has the same map tag and data_subset as what you've picked for the data and use all of them, unless you set an option to say otherwise.
The signal map directory must also contain a spectrum file containing the :math:`\ell(\ell+1)C_\ell/(2\pi)` spectra used to create the realizations in the directory.
This spectrum in used in computing the transfer function.

One mask is required per map tag.
These files begin with ``mask_map_`` instead of ``map_``.

Optional inputs are described in :ref:`Algorithm<Algorithm: Step by Step>`.

Non-Map Data
............

The other data you'll need to provide are your beam window functions and the band centers of the input maps (if fitting for foregrounds).
These are specified in a config file, an example of which is in `config_example.ini <https://github.com/annegambrel/xfaster/blob/main/example/config_example.ini>`_.

Beams can be specified either with a simple FHWM, if using a Gaussian beam model, or with an ell-by-ell beam window function, stored in a ``.npz`` file.
The ``.npz file`` should contain a dictionary with a key for each map tag.
The beams can be an :math:`\ell` -length vector, or a 3 :math:`\times \ell` - shape array if different beams are desired for Stokes I/Q/U.

Running the code
================

