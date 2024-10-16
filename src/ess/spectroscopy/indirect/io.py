# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import h5py
import numpy
import scippnexus
from scipp import DataArray

from ..types import NormWavelengthEvents, NXspeFileName, NXspeFileNames


def to_nxspe(events: NormWavelengthEvents, base: NXspeFileName) -> NXspeFileNames:
    """Take events, which have been binned in sloth and have monitor counts per bin,
    and output one NXspe file per setting

    Parameters
    ----------
    events: scipp.DataArray
        The events binned in (setting, event_id, incident_wavelength) with
        at least 'monitor', 'a3', and 'theta' bin coordinates.
        The events _and_ the bins must also have an 'incident_wavelength' coordinate.
    base: str | Path
        The filename base used to produce each NXspe filename.
    """
    from pathlib import Path

    from tqdm import tqdm

    dim = 'setting'
    length = len(str(events.sizes[dim] + 1))
    files = []
    if not isinstance(base, Path):
        base = Path(base)
    parent = base.parent
    if not parent.exists():
        parent.mkdir(parents=True)

    progress = tqdm(range(events.sizes[dim]))
    for i in progress:
        ev = events[dim, i]
        fn = str(base) + '_' + f'{i+1}'.rjust(length, '0') + '.nxspe'
        files.append(NXspeFileName(fn))
        _to_one_nxspe(ev, fn, progress)
    return NXspeFileNames(files)


def _make_group(group: h5py.Group) -> scippnexus.Group:
    return scippnexus.Group(group, definitions=scippnexus.base_definitions())


def named_coords_midpoint_broadcast(data, names):
    from scipp import midpoints

    sizes = data.sizes

    def coord_midpoint_broadcast(coord):
        for x in coord.dims:
            if x in sizes and coord.sizes[x] == 1 + sizes[x]:
                coord = midpoints(coord, x)
        return coord.broadcast(sizes=sizes)

    return {k: coord_midpoint_broadcast(data.coords[k]) for k in names}


def get_empty_centres(data, keep: list[str], graph: dict, extract: list[str], dim: str):
    from scipp import array

    # histogram first
    data = data.hist().transform_coords(keep, graph=graph)
    empty = (
        data.values == 0
    )  # as used below, equivalent to numpy.nonzero(data.values == 0)
    coords = named_coords_midpoint_broadcast(data, extract)
    for k, v in coords.items():
        coords[k] = array(
            values=v.values[empty], dims=[dim], unit=v.unit, dtype=v.dtype
        )
        if v.variances is not None:
            coords[k].variances = v.variances[empty]
    return coords


def update_empty_centres(data, first, empty):
    from numpy import cumsum

    begin = first + cumsum(empty)
    data.bins.constituents['begin'][empty] = begin
    data.bins.constituents['end'][empty] = begin + 1
    return data


def initialize_needed(sizes, coord):
    from scipp import DType, full

    dtype = coord.dtype
    if dtype == DType.float64 or dtype == DType.float32:
        from numpy import nan

        default = nan
    elif dtype == DType.int32 or dtype == DType.int64:
        default = -1
    elif dtype == DType.string:
        default = ""
    elif dtype == DType.bool:
        default = False
    elif dtype == DType.datetime64:
        from scipp import datetime

        default = datetime(0)
    else:
        default = -1
    return full(sizes=sizes, unit=coord.unit, dtype=dtype, value=default)


def _add_null_observations_rebin(events: DataArray, targets: list[str], graph: dict):
    # In the future it may be possible to do this without re-binning at the end
    # https://github.com/scipp/scipp/issues/1967#issuecomment-958680504
    from scipp import concat, zeros
    from scipp import max as scipp_max
    from scipp import min as scipp_min

    needed = events.dims
    orig = events.bins.constituents['data']
    dim = orig.dims[0]
    if any(n not in orig.coords for n in needed):
        bin_begin = events.bins.constituents['begin'].values.flatten()
        bin_end = events.bins.constituents['end'].values.flatten()
        for n in needed:
            t = initialize_needed(orig.sizes, events.coords[n])
            for v, b, e in zip(
                events.coords[n].values.flatten(), bin_begin, bin_end, strict=False
            ):
                t.values.flat[b:e] = v
            orig.coords[n] = t

    extract = targets + [n for n in needed if n not in targets]
    coords = get_empty_centres(events, targets, graph, extract, dim)
    rows = max(coords[target].sizes[dim] for target in targets)
    nulls = DataArray(
        zeros(sizes={dim: rows}, unit='counts', dtype=orig.dtype), coords=coords
    )
    if events.variances is not None:
        nulls.variances = 1 + nulls.values

    # extend the list of events
    # FIXME Why does this give different bin means than the event list without zeros?
    comb = concat((orig, nulls), dim)

    # then bin or combine the coordinates of the input events
    binned = {
        k: events.coords[k]
        for k in extract
        if k in events.coords and events.coords[k].sizes[k] == 1 + events.sizes[k]
    }
    grouped = [k for k in extract if k in events.coords and k not in binned]
    # any other extract entries are left on the events' coordinates

    out = comb
    for group in grouped:
        smallest = scipp_min(events.coords[group])
        largest = scipp_max(events.coords[group])
        largest.value += 1  # label based slicing is upper-bound exclusive :(
        out = out.group(group)[group, smallest:largest]
    out = out.bin(binned)
    for k, v in events.coords.items():
        out.coords[k] = v
    return out


def _add_null_observations_append(events: DataArray, targets: list[str], graph: dict):
    """Extend the events list after finding the number of empty bins. Then update the
    bin-indexing for the empty bins to point at the added end-of-list null events

    If done correctly, the input event information is undisturbed and the null events
    get assigned to bins which were otherwise unused.
    """
    from numpy import arange
    from scipp import concat, zeros

    input_event_list = events.bins.constituents['data']
    needed = events.dims
    dim = input_event_list.dims[0]
    extract = targets + [n for n in needed if n not in targets]
    coords = get_empty_centres(events, targets, graph, extract, dim)
    rows = max(coords[target].sizes[dim] for target in targets)
    nulls = DataArray(
        zeros(sizes={dim: rows}, unit='counts', dtype=input_event_list.dtype),
        coords=coords,
    )
    if events.variances is not None:
        nulls.variances = 1 + nulls.values
    # make the new event list by concatenating the null observations on the end
    output_event_list = concat((input_event_list, nulls), dim=dim)

    # FIXME Something about the following scrambles the returned binned data?
    # the first index for the _newly added_ null observations in the event list
    first = input_event_list.sizes[dim]
    begin = first + arange(rows)
    empty = (
        events.bins.constituents['begin'].values
        == events.bins.constituents['end'].values
    )
    events.bins.constituents['begin'].values[empty] = begin
    events.bins.constituents['end'].values[empty] = begin + 1
    events.bins.constituents['data'] = output_event_list
    return events


#
# def _add_null_observations(events: DataArray, targets: list[str], graph: dict):
#     # This leads to wrong intensities for some reason?
#     # return _add_null_observations_rebin(events, targets, graph)
#     return _add_null_observations_append(events, targets, graph)


def _clean_up_observations(events: DataArray, keep: list[str]):
    coords = [x for x in events.bins.coords if x not in keep]
    for coord in coords:
        del events.bins.coords[coord]
    return events


def _to_one_nxspe(events: DataArray, filename: str, progress):
    """Use scippnexus to create the NXspe file"""
    from scipp import full, scalar
    from scipp.constants import Planck, neutron_mass
    from scippnexus import (
        NXcollection,
        NXdata,
        NXdetector,
        NXentry,
        NXfermi_chopper,
        NXinstrument,
        NXsample,
    )

    def lambda_to_ei(incident_wavelength):
        return ((Planck / incident_wavelength) ** 2 / neutron_mass / 2).to(unit='meV')

    def ei_ef_to_en(incident_energy, final_energy):
        return incident_energy - final_energy

    observations = events.copy()
    ef = events.coords['final_energy']
    # FIXME somehow this multiplication allows the function to run, otherwise
    #       _something_ breaks and the Python kernel dies. Maybe dtype shenanigans?
    # ki_over_kf = sqrt(events.bins.coords['incident_energy'] / ef)
    # observations *= ki_over_kf

    # Adding null events requires replicating all coordinates of real events.
    # We don't necessarily use all present event coordinates, so remove any we won't use
    targets = [
        'energy_transfer',
        'incident_energy',
        'incident_wavelength',
        'final_energy',
    ]
    observations = _clean_up_observations(observations, targets)
    # And provide a means to calculate the rest from bin (not event) information
    # we are sure incident_wavelength is a bin coordinate already, so can skip it.
    graph = {
        'incident_energy': lambda_to_ei,
        'energy_transfer': ei_ef_to_en,
    }
    # FIXME adding zero-weight observations doesn't work right. Why?
    # observations = _add_null_observations(observations, targets, graph)

    # combine the per bin intensities and normalize by monitor counts
    # Note, applying this normalization to the _events_ would require splitting
    # the bin monitor counts between the bin events.
    normalize_by = events.coords['monitor']  # .broadcast(sizes=observations.sizes)
    if normalize_by.variances is not None:
        # we need to ignore the monitor uncertainty for the time being
        normalize_by = normalize_by.copy()
        normalize_by.variances = None

    # # ensure that we have the _event_ averages for the needed coordinates:
    # #   with bins.mean energy_transfer around zero became NaN??
    # averages = {t: observations.bins.coords[t].bins.nanmean() for t in targets}
    # throw away all other event information
    # plus ensure the quantities we need get calculated from the bin coordinates
    observations = observations.hist().transform_coords(targets, graph=graph)
    # assign the average values of events, where they exist, to the bin coordinates
    # for t in averages:
    #     a = averages[t].transpose(observations.coords[t].dims)
    #     non_nan = isfinite(a.values)
    #     observations.coords[t].values[non_nan] = a.values[non_nan]
    # progress.set_description(f'averages {list(averages)}')

    progress.set_description(f"<E> = {observations.coords['energy_transfer'].mean():c}")

    if observations.variances is None:
        observations.variances = observations.values  # correct for counting statistics
        observations.variances[observations.values == 0] = 1

    # observations.data = observations.data / normalize_by

    # Some things don't actually get written to the file if I don't copy first... why?
    psi = observations.coords['a3']
    polar = observations.coords['theta']
    azimuthal = 0 * polar
    azimuthal_width = azimuthal + scalar(2.0, unit='degree')
    polar_width = azimuthal + scalar(0.1, unit='degree')
    distance = full(sizes=polar.sizes, unit='m', value=3.0, dtype=polar.dtype)
    data = observations.data.copy()
    error = 0 * observations.data.values
    if observations.data.variances is not None:
        observations.data.variances[observations.data.variances == 0] = 1
        error = numpy.sqrt(observations.data.variances)
    energy_transfer = observations.coords['energy_transfer'].copy()
    final_energy = observations.coords['final_energy'].copy()

    with h5py.File(filename, mode='w') as f:
        # make / in the file
        root = _make_group(f)
        # it _must_ contain an NXentry group, called [anything which is allowed]?
        # with two fields and five subgroups required
        entry = root.create_class('entry', NXentry)
        # the name of  the author program
        entry.create_field('program_name', scalar('essspectroscopy'))
        # and the NXDL schema information -- currently version 3.1
        definition = entry.create_field('definition', scalar('NXSPE'))
        definition.attrs['version'] = '3.1'

        # the entry group also contains five subgroups

        # the NXcollection group must contain three fields
        nxinfo = entry.create_class('NXSPE_info', NXcollection)
        nxinfo.create_field('fixed_energy', ef)
        nxinfo.create_field('ki_over_kf_scaling', scalar(True))
        nxinfo.create_field('psi', psi)

        # the NXdata group has 8 required fields
        nxdata = entry.create_class('data', NXdata)
        nxdata.create_field('azimuthal', azimuthal)
        nxdata.create_field('azimuthal_width', azimuthal_width)
        nxdata.create_field('polar', polar)
        nxdata.create_field('polar_width', polar_width)
        nxdata.create_field('distance', distance)
        nxdata.create_field('data', data)
        nxdata.create_field('error', error)
        nxdata.create_field('energy', energy_transfer)

        # the NXinstrument group has one required field and one required group
        instrument = entry.create_class('instrument', NXinstrument)
        instrument.create_field('name', scalar('SIMBIFROST'))
        fermi = instrument.create_class('fermi_chopper', NXfermi_chopper)
        fermi.create_field('energy', scalar(numpy.nan, unit='meV'))
        detectors = instrument.create_class('detectors', NXdetector)
        detectors.create_field('final_energy', final_energy)

        # and the NXsample group has three required fields
        sample = entry.create_class('sample', NXsample)
        sample.create_field('rotation_angle', psi)
        sample.create_field('seblock', scalar(""))
        sample.create_field('temperature', scalar(numpy.nan, unit='K'))


providers = (to_nxspe,)
