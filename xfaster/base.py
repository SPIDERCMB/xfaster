from __future__ import absolute_import
from __future__ import print_function
import inspect
from configparser import RawConfigParser as rcp

__all__ = ["XFasterConfig", "get_func_defaults", "extract_func_kwargs"]


class XFasterConfig(rcp):
    """
    ConfigParser subclass for storing command line options and config.
    """

    def __init__(self, defaults=None, default_sec="Uncategorized"):
        """
        Class that tracks command-line options for storage to disk.

        Arguments
        ---------
        defaults : dict
            Dictionary of overall configuration values.
            Eg: locals() at beginning of function, or vars(args) from argparse
        default_sec : string, optional
            The name of the default section in the configuration file.
        """
        from collections import OrderedDict

        super(XFasterConfig, self).__init__(dict_type=OrderedDict)
        self.default_sec = default_sec
        self.add_section(default_sec)
        if defaults is not None:
            self.update(defaults)

    def update(self, options, section=None):
        """
        Update configuration options with a dictionary. Behaves like
        dict.update() for specified section but also clears options of the same
        name from the default section.

        Arguments
        ---------
        options : dict
            The options to update
        section : string, optional
            Name of section to update. Default: self.default_sec
        """
        if section is None:
            section = self.default_sec
        if not self.has_section(section):
            self.add_section(section)
        # change kwargs to be like any other options
        kw = options.pop("kwargs", None)
        if isinstance(kw, dict):
            options.update(kw)
        for k, v in sorted(options.items()):
            self.remove_option(self.default_sec, k)
            self.set(section, k, str(v))

    def sort(self):
        """
        Sort the items in each section of the configuration.
        """
        for section, section_items in self.items():
            if sorted(section_items) == list(section_items):
                continue

            section_dict = {k: v for k, v in section_items.items()}

            for k in list(section_items):
                self.remove_option(section, k)

            for k, v in sorted(section_dict.items()):
                self.set(section, k, v)

    def write(self, fp=None, sort=True):
        """
        Write an .ini-format representation of the configuration state.
        Keys are stored alphabetically if `sort` is True.

        Arguments
        ---------
        fp : file object
            If None, write to `sys.stdout`.
        sort : bool
            If True, sort items in each section alphabetically.
        """
        if fp is None:
            import sys

            fp = sys.stdout

        if sort:
            self.sort()
        super(XFasterConfig, self).write(fp)


def get_func_defaults(func):
    """
    Return a dictionary containing the default values for each keyword
    argument of the given function

    Arguments
    ---------
    func : function or callable
        This function's keyword arguments will be extracted.

    Returns
    -------
    dict of kwargs and their default values
    """
    spec = inspect.getargspec(func)
    from collections import OrderedDict

    return OrderedDict(zip(spec.args[-len(spec.defaults) :], spec.defaults))


def extract_func_kwargs(func, kwargs, pop=False, others_ok=True, warn=False):
    """
    Extract arguments for a given function from a kwargs dictionary

    Arguments
    ---------
    func : function or callable
        This function's keyword arguments will be extracted.
    kwargs : dict
        Dictionary of keyword arguments from which to extract.
        NOTE: pass the kwargs dict itself, not **kwargs
    pop : bool, optional
        Whether to pop matching arguments from kwargs.
    others_ok : bool
        If False, an exception will be raised when kwargs contains keys
        that are not keyword arguments of func.
    warn : bool
        If True, a warning is issued when kwargs contains keys that are not
        keyword arguments of func.  Use with `others_ok=True`.

    Returns
    -------
    kwargs : dict
        Dict of items from kwargs for which func has matching keyword arguments
    """
    spec = inspect.getargspec(func)
    func_args = set(spec.args[-len(spec.defaults) :])
    ret = {}
    for k in list(kwargs.keys()):
        if k in func_args:
            if pop:
                ret[k] = kwargs.pop(k)
            else:
                ret[k] = kwargs.get(k)
        elif not others_ok:
            msg = "Found invalid keyword argument: {}".format(k)
            raise TypeError(msg)
    if warn and kwargs:
        s = ", ".join(kwargs.keys())
        warn("Ignoring invalid keyword arguments: {}".format(s), Warning)
    return ret
