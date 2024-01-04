"""just miscellanous functions copied over from tbrugere/ml_lib/misc"""

import functools as ft
from logging import info
from typing import Sequence
from time import perf_counter

def auto_repr(*fields: str):
    """
    decorator.

    Creates an automatic __repr__ function for a class. useful for debugging since printing
    Args:
        fields: the fields to display in that repr
    """

    def repr(self, fields: Sequence[str]):
        return f"""{self.__class__.__name__}({
            ', '.join(f"{field}= {getattr(self, field)}" for field in fields)
            })"""

    def decorator(cls):
        cls.__repr__ = ft.partialmethod(repr, fields=fields)
        return cls

    return decorator


def all_equal(*args):
    """
    tests whether all the passed arguments are equal. 
    useful for checking dimensions for a lot of vectors for ex
    """
    match args:
        case ():
            return True
        case (x0, *rest):
            return all(i == x0 for i in rest)

old_time = perf_counter()
def debug_time(out_string=None, log=info):
    """only works if is_debug is True
    If out_string is true, prints out_string as well as the time elapsed since the last call to debug_time

    Args:
      out_string: (Default value = None)
      log: what function to use to log (Default value = logging.info)

    Returns:

    """
    global old_time
    if out_string is None:
        old_time = perf_counter()
        return
    new_time = perf_counter()
    log(f"{out_string}: time elapsed {new_time - old_time}s")
    old_time = new_time
