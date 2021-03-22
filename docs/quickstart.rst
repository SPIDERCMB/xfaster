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

The code requires a certain directory structure for your input maps.

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
