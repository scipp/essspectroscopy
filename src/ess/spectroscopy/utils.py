# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import scipp as sc


def _elem_unit(var: sc.Variable) -> sc.Unit:
    return var.bins.unit if var.bins is not None else var.unit


def to_unit_of(variable: sc.Variable, to: sc.Variable) -> sc.Variable:
    """Convert the input to the unit of another variable."""
    target = _elem_unit(to)
    unit = _elem_unit(variable)
    if target is None and unit is None:
        return variable
    if target is None or unit is None:
        raise ValueError(f"Can not find the units to use for {variable} from {to}")
    return variable.to(unit=target, copy=False)


def range_normalized(variable: sc.Variable, minimum: sc.Variable, maximum: sc.Variable):
    """Convert a variable to a normalized range

    Parameters
    ----------
    variable:
        The values to be normalized
    minimum:
        The minimal value that the input `variable` could take
    maximum:
        The maximal value that the input `variable` could take

    Returns
    -------
    :
        The input `variable` rescaled by the range of allowed values
    """
    full = maximum - minimum
    minimum, full = (to_unit_of(x, to=variable) for x in (minimum, full))
    return variable / full - minimum / full


def is_in_coords(x: sc.DataArray, name: str):
    return name in x.coords or (x.bins is not None and name in x.bins.coords)
