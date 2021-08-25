import os

from setuptools import setup


def read(rel_path):
    # type: (str) -> str
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path):
    # type: (str) -> str
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name="xfaster",
    version=get_version("xfaster/__init__.py"),
    description="A fast power spectrum and likelihood estimator for CMB datasets",
    long_description=read("README.rst"),
    license="MIT",
    url="https://github.com/SPIDER-CMB/xfaster",
    packages=["xfaster"],
    package_dir={"xfaster": "xfaster"},
    entry_points={"console_scripts": ["xfaster=xfaster:xfaster_main"]},
    install_requires=["numpy>1.17.5", "healpy", "camb", "emcee", "h5py"],
    python_requires='>=3.0',
)
