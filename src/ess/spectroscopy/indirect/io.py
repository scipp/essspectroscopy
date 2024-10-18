# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import h5py
import numpy
import scippnexus
from scipp import DataArray

from ..types import NormWavelengthEvents, NXspeFileName, NXspeFileNames


def to_nxspe(events: NormWavelengthEvents, base: NXspeFileName) -> NXspeFileNames:
    """Take events, which have been binned in incident wavelength and have monitor
    counts per bin, and output one NXspe file per setting

    Parameters
    ----------
    events: scipp.DataArray
        The events binned in (setting, event_id, incident_wavelength) with
        at least 'monitor', 'a3', and 'theta' bin coordinates.
        The events and the bins must also have an 'incident_wavelength' coordinate.
    base: str | Path
        The filename base used to produce each NXspe filename.

    Returns
    -------
    :
        The list of filenames containing the NXspe data. If there are N settings in the
        input DataArray, N filenames are returned. The names are of the form
            {base}_{i+1:0{ceil(log10(N+1))}d}.nxspe
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


def _lambda_to_ei(incident_wavelength):
    from scipp.constants import Planck, neutron_mass

    return ((Planck / incident_wavelength) ** 2 / neutron_mass / 2).to(unit='meV')


def _ei_ef_to_en(incident_energy, final_energy):
    return incident_energy - final_energy


def _to_one_nxspe(events: DataArray, filename: str, progress):
    """Use scippnexus to create the NXspe file"""
    from scipp import full, scalar, sqrt
    from scippnexus import (
        NXcollection,
        NXdata,
        NXentry,
        NXfermi_chopper,
        NXinstrument,
        NXsample,
    )

    observations = events.copy()
    ef = events.coords['final_energy']
    observations *= sqrt(events.bins.coords['incident_energy'] / ef)

    # Adding null events requires replicating all coordinates of real events.
    # We don't necessarily use all present event coordinates, so remove any we won't use
    targets = [
        'energy_transfer',
        'incident_energy',
        'incident_wavelength',
        'final_energy',
    ]
    for coord in [x for x in events.bins.coords if x not in targets]:
        del observations.bins.coords[coord]
    # And provide a means to calculate the rest from bin (not event) information
    # we are sure incident_wavelength is a bin coordinate already, so can skip it.
    graph = {
        'incident_energy': _lambda_to_ei,
        'energy_transfer': _ei_ef_to_en,
    }
    # # Adding zero-weight observations, i.e., null events, works now, but involves
    # # a memory-copy of the data array and its bin structure. Since it isn't strictly
    # # necessary for exporting NXspe, it can be skipped for now.
    # observations = add_null_observations(observations, targets, graph)

    # combine the per bin intensities and normalize by monitor counts
    # Note, applying this normalization to the _events_ would require splitting
    # the bin monitor counts between the bin events.
    normalize_by = events.coords['monitor']  # .broadcast(sizes=observations.sizes)
    if normalize_by.variances is not None:
        # we need to ignore the monitor uncertainty for the time being
        normalize_by = normalize_by.copy()
        normalize_by.variances = None

    observations = observations.hist().transform_coords(targets, graph=graph)

    if observations.variances is None:
        observations.variances = observations.values  # correct for counting statistics
        observations.variances[observations.values == 0] = 1
    if observations.data.variances is not None:
        observations.data.variances[observations.data.variances == 0] = 1
    # the transform_coords above renamed the 'incident_wavelength' dimension
    # to 'energy_transfer' ... so we need to do the same to normalize_by or else
    # scipp tries to broadcast when it doesn't need to.
    observations.data = observations.data / normalize_by.rename_dims(
        incident_wavelength='energy_transfer'
    )

    # Some things don't actually get written to the file if I don't copy first... why?
    psi = observations.coords['a3']
    polar = observations.coords['theta']
    azimuthal = 0 * polar  # all detectors are in the horizontal plane

    azimuthal_width = azimuthal + scalar(2.0, unit='degree')
    polar_width = azimuthal + scalar(0.1, unit='degree')
    distance = full(sizes=polar.sizes, unit='m', value=3.0, dtype=polar.dtype)
    data = observations.data.copy()
    error = 0 * observations.data.values
    if observations.data.variances is not None:
        error = numpy.sqrt(observations.data.variances)
    energy_transfer = observations.coords['energy_transfer'].copy()
    incident_energy = observations.coords['incident_energy'].copy()
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
        nxinfo.create_field('fixed_energy', final_energy)
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
        # Actually more useful extensions to NXspe for an instrument like BIFROST
        nxdata.create_field('final_energy', final_energy)
        nxdata.create_field('incident_energy', incident_energy)

        # the NXinstrument group has one required field and one required group
        instrument = entry.create_class('instrument', NXinstrument)
        instrument.create_field('name', scalar('SIMBIFROST'))
        fermi = instrument.create_class('fermi_chopper', NXfermi_chopper)
        fermi.create_field('energy', scalar(numpy.nan, unit='meV'))

        # and the NXsample group has three required fields
        sample = entry.create_class('sample', NXsample)
        sample.create_field('rotation_angle', psi)
        sample.create_field('seblock', scalar(""))
        sample.create_field('temperature', scalar(numpy.nan, unit='K'))


providers = (to_nxspe,)