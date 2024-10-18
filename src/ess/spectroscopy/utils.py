# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from scipp import DataArray, Variable


def in_same_unit(b: Variable, to: Variable | None = None) -> Variable:
    def unit(x):
        if x.bins is not None:
            return x.bins.unit
        return x.unit

    if to is None:
        raise ValueError("The to unit-full object must be specified")

    a_unit = unit(to)
    b_unit = unit(b)
    if a_unit is None and b_unit is None:
        return b
    if a_unit is None or b_unit is None:
        raise ValueError(f"Can not find the units to use for {b} from {to}")
    if a_unit != b_unit:
        b = b.to(unit=a_unit)
    return b


def is_in_coords(x: DataArray, name: str):
    return name in x.coords or (x.bins is not None and name in x.bins.coords)


def named_coords_midpoint_broadcast(
    data: DataArray, names: list[str]
) -> dict[str, Variable]:
    """Find the singular value for the named coordinates for all bins

    Bin coordinates can represent bin-edges or singular values.
    Bin edges can be converted to singular values by the scipp free function
    `midpoints`, but this function can not be used indiscriminately because it
    will happily calculate midpoint values from singular bin values.

    Parameters
    ----------
    data: scipp.DataArray
        Ideally a histogram DataArray, may also work for event data.
    names: list[str]
        The coordinate names to convert to singular bin values and broadcast to full
        rank for the shape of data.

    Returns
    -------
    :
        A dictionary of the coordinate names and their full-rank singular-value arrays
    """
    from scipp import midpoints

    sizes = data.sizes

    def coord_midpoint_broadcast(coord):
        for x in coord.dims:
            if x in sizes and coord.sizes[x] == 1 + sizes[x]:
                coord = midpoints(coord, x)
        return coord.broadcast(sizes=sizes)

    return {k: coord_midpoint_broadcast(data.coords[k]) for k in names}


def get_empty_centres(
    data: DataArray, keep: list[str], graph: dict, extract: list[str], dim: str
) -> dict[str, Variable]:
    """Return the coordinates of bin centers which contain no events

    Parameters
    ----------
    data: scipp.DataArray
        Ideally indexing event data, but must support use of its hist method
    keep: list[str]
        Coordinate names to obtain from the bin coordinates of data, possibly after
        transformation
    graph: dict[str, str | callable]
        Transformation graph to be used with data.hist().transform_coords to calculate
        or otherwise transform bin coordinates to obtain the named keep coordinates
    extract: list[str]
        Coordinate names to extract from the transformed coordinates as bin averages
        if they represent edges, or singular values if not. This list likely includes
        all entries in `keep` and at least a subset of the bin coordinates of the
        provided data array
    dim: name
        The dimension to concatenate each of the coordinate value array along, this
        is the dimension of the resulting event-like list.

    Returns
    -------
    :
        An list of coordinates like an event list, representing the average bin
        values for the `extract` coordinates for all bins which contain no events
    """
    from scipp import array

    # Identify which bins have no events before histogramming in case there are
    # bins with sum() == 0 but size() > 0.
    empty = data.bins.size().values == 0  # logical array good enough for indexing
    # histogram first
    data = data.hist().transform_coords(keep, graph=graph)
    coords = named_coords_midpoint_broadcast(data, extract)
    for k, v in coords.items():
        coords[k] = array(
            values=v.values[empty], dims=[dim], unit=v.unit, dtype=v.dtype
        )
        if v.variances is not None:
            coords[k].variances = v.variances[empty]
    return coords


def add_null_observations(
    events: DataArray, targets: list[str], graph: dict
) -> DataArray:
    """Make null events for observations without events

    Extends the input events list with zero-intensity events that have the bin average
    value of any target coordinates plus the binning dimensions of the input array.

    Parameters
    ----------
    events: scipp.DataArray
        A binned scipp.DataArray -- such that `events.bins` is not None
    targets: list[str]
        Any coordinate needed for the null observations which should be calculated
        from the bin coordinates, e.g., if binned in a coordinate 'x' and  grouped
        in 'y', then null observations likely need the mean 'x' value of their bin
        and the single 'y' value of their bin.
    graph: dict[str: str|callable]
        The transformation graph for events.hist().transform_coords which can provide
        a value for any `target` that is not a bin coordinate; e.g., the bin coordinates
        may contain 'incident_energy' and 'final_energy', and we may require the target
        'energy_transfer', so graph must have an entry like
            {'energy_transfer':
                lambda incident_energy, final_energy: incident_energy - final_energy
            }

    Returns
    -------
    :
        A copy of the input events list with null observations appended to its end
        with a new wrapping scipp.bins and scipp.DataArray to include those null
        observations in the bin data.
    """
    from numpy import arange
    from scipp import bins, concat, zeros

    event_list = events.bins.constituents['data']
    needed = events.dims
    dim = event_list.dims[0]
    extract = targets + [n for n in needed if n not in targets]
    coords = get_empty_centres(events, targets, graph, extract, dim)
    rows = max(coords[target].sizes[dim] for target in targets)
    nulls = DataArray(
        zeros(sizes={dim: rows}, unit='counts', dtype=event_list.dtype),
        coords=coords,
    )
    if events.variances is not None:
        nulls.variances = 1 + nulls.values
    # make the new event list by concatenating the null observations on the end
    observation_list = concat((event_list, nulls), dim=dim)

    # We can _modify_ the `events.bins.constituents` arrays in place, but we
    # CAN NOT _replace_ the arrays!
    # Instead, we must create a new bins object, with new begin and end arrays to avoid
    # creating out-of-bounds array indexes for the input binned array.
    begin = events.bins.constituents['begin'].copy()
    end = events.bins.constituents['end'].copy()
    empty = begin.values == end.values
    begin.values[empty] = event_list.sizes[dim] + arange(rows)
    end.values[empty] = 1 + begin.values[empty]
    observations = DataArray(bins(begin=begin, end=end, dim=dim, data=observation_list))
    for k, v in events.coords.items():
        observations.coords[k] = v
    return observations
