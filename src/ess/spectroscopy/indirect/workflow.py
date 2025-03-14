# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import scipp as sc
import scippnexus as snx
from loguru import logger
from scipp import Variable

from ..types import (
    Analyzers,
    Choppers,
    DetectorData,
    Filename,
    InstrumentAngles,
    MonitorData,
    NeXusDetectorName,
    NeXusMonitorName,
    NormWavelengthEvents,
    NXspeFileName,
    Position,
    PreopenNeXusFile,
    SampleEvents,
    SampleRun,
    SourceSamplePathLength,
    TimeOfFlightLookupTable,
    TofSampleEvents,
)

PIXEL_NAME = 'detector_number'


def _load_all(group, obj_type):
    """Helper to find and load all subgroups of a specific scippnexus type"""
    return {name: obj[...] for name, obj in group[obj_type].items()}


def _load_named(group, obj_type, names):
    """Helper to find and load all subgroups of a specific scippnexus type with
    group name in an allowed set"""
    return {name: obj[...] for name, obj in group[obj_type].items() if name in names}


def ess_source_frequency():
    """Input for a sciline workflow, returns the ESS source frequency of 14 Hz"""
    from scipp import scalar

    return scalar(14.0, unit='Hz')


def ess_source_period():
    """Input for a sciline workflow, returns the ESS source period of 1/(14 Hz)"""
    return (1 / ess_source_frequency()).to(unit='ns')


def ess_source_delay():
    """Input for a sciline workflow, returns per-wavelength source delays of 0 s"""
    from scipp import array

    return array(values=[0, 0.0], dims=['wavelength'], unit='sec', dtype='float64')


def ess_source_duration():
    """Input for a sciline workflow, returns source pulse duration of 3 msec"""
    from scipp import scalar

    return scalar(3.0, unit='msec').to(unit='sec')


def ess_source_velocities():
    """Input for a sciline workflow, returns per-wavelength source velocity limits

    Notes
    -----
    The chosen limits are not based on any properties of the source, but rather
    entirely on the equivalent energy range, which is chosen to be
    ~53 micro electron volts to 530 milli electron volts. This energy range should
    be sufficient for all intended incident energies of the ESS spectrometer suite,
    but may not be sufficient to capture spurious energies that pass through
    the real instruments.

    Returns
    -------
    :
        A 1-D scipp Variable with values=[100, 10000] m/s
    """
    from scipp import array

    return array(values=[100, 1e4], dims=['wavelength'], unit='m/s')


def convert_simulated_time_to_event_time_offset(data):
    """Helper to make McStas simulated event data look more like real data

    McStas has the ability to track the time-of-flight from source to detector for
    every probabilistic neutron ray. This is very helpful, but unfortunately real
    instrument at ESS are not able to record the same information due to how the
    timing and data collection systems work.

    Real neutron events will record their event_time_zero most-recent-pulse reference
    time, and their event_time_offset detection time relative to that reference time.
    These two values added together give a real wall time; and information about the
    primary spectrometer is necessary to find any time-of-flight

    This function takes event data with per-event coordinate event_time_offset
    (actually McStas time-of-arrival) and converts the coordinate to be
    the time-of-arrival modulo the source repetition period.

    Notes
    -----
    If the input data has realistic event_time_offset values, this function should
    be a noop.

    Returns
    -------
    :
        A copy of the data with realistic per-event coordinate event_time_offset.
    """
    res = data.transform_coords(
        frame_time=lambda event_time_offset: event_time_offset % ess_source_period(),
        rename_dims=False,
        keep_intermediate=False,
        keep_inputs=False,
    )
    return res.transform_coords(event_time_offset='frame_time', keep_inputs=False)


def analyzer_per_detector(analyzers: list[str], triplets: list[str]) -> dict[str, str]:
    """Find the right analyzer name for each detector

    Notes
    -----
    Depends heavily on the names of components being preceded by an in-instrument index,
    and the analyzer and detector components being separated in index by 2.
    If either condition changes this function will need to be modified.

    Parameters
    ----------
    analyzers: list[str]
        The names of analyzer components, typically generated from the keys of a dict
    triplets: dict
        The names of triplet detector components, typically generated from the
        keys of a dict

    Returns
    -------
    :
        A dictionary with detector name keys and their associated analyzer name values
    """

    # TODO Update this function if the NeXus structure changes.
    def correct_index(d, a):
        detector_index = int(d.split('_', 1)[0])
        analyzer_index = detector_index - 2
        return a.startswith(str(analyzer_index))

    return {d: next(x for x in analyzers if correct_index(d, x)) for d in triplets}


def detector_per_pixel(triplets: dict) -> dict[int, str]:
    """Find the right detector name for every pixel index

    Parameters
    ----------
    triplets: dict[str, scipp.DataGroup]
        A mapping of detector component name to a loaded scippnexus.NXdetector group,
        with 'data' group member that has a 'detector_number' coordinate

    Returns
    -------
    :
        The mapping of detector_number to detector component name
    """
    return {
        i: name
        for name, det in triplets.items()
        for i in det.coords['detector_number'].values.flatten()
    }


def combine_analyzers(analyzers: dict, triplets: sc.DataGroup[sc.DataArray]):
    """Combine needed analyzer properties into a single array,
    duplicating information, to have per-pixel data

    BIFROST has 45 analyzers and 45 triplet detectors, each with some number of pixels,
    N. Calculations for the properties of neutrons which make it to each detector pixel
    need per-pixel data about the analyzer associated with that pixel. This function
    collects the required data and combines it into a single array.

    Analyzer information required to determine the secondary-spectrometer neutron
    properties are the center-of-mass position of the analyzer, the orientation of
    the analyzer, and the lattice spacing separating crystal planes in the analyzer.

    Notes
    -----
    Since there are N pixels per detector, the returned array is strictly N-times
    larger than necessary, but the optimization to use only minimal information is
    left for the future

    Parameters
    ----------
    analyzers: dict[str, scipp.DataGroup]
        Maps analyzer component name to loaded scippnexus.NXcrystal group
    triplets: dict[str, scipp.DataGroup]
        Maps detector component name to loaded scippnexus.NXdetector group

    Returns
    -------
    :
        A single array with pixel dimension and the per-pixel analyzer information
    """
    from scipp import Dataset, array, concat
    from scippnexus import compute_positions

    def analyzer_extract(obj):
        obj = compute_positions(obj, store_transform='transform')
        return Dataset(data={k: obj[k] for k in ('position', 'transform', 'd_spacing')})

    extracted = {k: analyzer_extract(v) for k, v in analyzers.items()}
    d2a = analyzer_per_detector(list(analyzers.keys()), list(triplets.keys()))
    p2d = detector_per_pixel(triplets)

    p2a = {k: extracted[d2a[v]] for k, v in p2d.items()}
    pixels = sorted(p2a)
    data = concat([p2a[p] for p in pixels], dim=PIXEL_NAME)
    data[PIXEL_NAME] = array(values=pixels, dims=[PIXEL_NAME], unit=None)
    return data


def combine_detectors(triplets: sc.DataGroup[sc.DataArray]):
    """Combine needed detector properties into a single array

    BIFROST has 45 analyzers and 45 triplet detectors, each with some number of pixels,
    N. Calculations for the properties of neutrons which make it to each detector pixel
    need per-pixel data. This function collects the required data and combines it into
    a single array.

    Detector information required to determine the secondary-spectrometer neutron
    properties are the center-of-mass position of each pixel.

    Parameters
    ----------
    triplets: dict[str, scipp.DataGroup]
        Maps detector component name to loaded scippnexus.NXdetector group

    Returns
    -------
    :
        A single array with pixel dimension and the per-pixel center of mass position
    """
    from scipp import Dataset, concat, sort

    def extract(obj):
        pixels = obj.coords['detector_number']
        midpoints = obj.coords['position']
        return Dataset(data={PIXEL_NAME: pixels, 'position': midpoints})

    data = concat([extract(v) for v in triplets.values()], dim='arm')
    data = Dataset({k: v.flatten(to=PIXEL_NAME) for k, v in data.items()})
    return sort(data, data[PIXEL_NAME].data)


def find_sample_detector_flight_time(sample, analyzers, detector_positions):
    """Use sciline to find the sample to detector flight time per detector pixel"""
    from sciline import Pipeline

    from ..types import (
        AnalyzerOrientation,
        AnalyzerPosition,
        DetectorPosition,
        ReciprocalLatticeSpacing,
        SampleDetectorFlightTime,
        SamplePosition,
    )
    from .kf import providers as kf_providers

    params = {
        SamplePosition: sample['position'],
        AnalyzerPosition: analyzers['position'].data,
        DetectorPosition: detector_positions,  # detectors['position'].data,
        AnalyzerOrientation: analyzers['transform'].data,
        ReciprocalLatticeSpacing: 2 * np.pi / analyzers['d_spacing'].data,
    }
    return params, Pipeline(kf_providers, params=params).get(
        SampleDetectorFlightTime
    ).compute().to(unit='ms')


def get_triplet_events(triplets: Iterable[sc.DataArray]):
    """Extract and combine the events from loaded scippneutron.NXdetector groups

    Parameters
    ----------
    triplets:
        An iterable container of loaded NXdetector groups, each with a 'data' member
        which contains the pixel data -- possibly multiple detector-specific
        (but consistent) dimensions -- with a coordinate identifying the
        'detector_number'

    Returns
    -------
    :
       The events from each triplet concatenated and sorted by the 'detector_number'
    """
    from scipp import concat, sort

    events = concat(list(triplets), dim='arm').flatten(to=PIXEL_NAME)
    events = sort(events, events.coords['detector_number'])
    return events


def get_sample_events(triplet_events: sc.DataArray, sample_detector_flight_times):
    """Return the events with the frame_time coordinate offset to time at the sample"""
    events = triplet_events.drop_coords(
        ('position', 'x_pixel_offset', 'y_pixel_offset')
    )
    events.bins.coords['event_time_offset'] -= sample_detector_flight_times.to(
        unit='ns'
    )
    events.bins.coords['event_time_offset'] %= ess_source_period()
    return events


def get_unwrapped_sample_events(
    filename,
    source_name,
    sample_name,
    sample_events,
    tof_lookup_table,
):
    """Shift frame_time at sample events to time-since-pulse events at sample"""
    from sciline import Pipeline

    from ..types import (
        Filename,
        SampleName,
        SourceName,
    )
    from .ki import providers as ki_providers

    params = {
        Filename: filename,
        SampleName: sample_name,
        SourceName: source_name,
        SampleEvents: sample_events,
        TimeOfFlightLookupTable: tof_lookup_table,
    }
    pipeline = Pipeline(ki_providers, params=params)

    events = pipeline.compute(TofSampleEvents)
    events = events.bins.drop_coords(('event_time_zero', 'event_time_offset'))
    return params, events, None  # primary


def get_unwrapped_monitor(
    filename, monitor: sc.DataArray, source_name, tof_lookup_table
) -> sc.DataArray:
    from sciline import Pipeline

    from ..types import (
        Filename,
        FrameTimeMonitor,
        MonitorPosition,
        SourceName,
        TofMonitor,
    )
    from .ki import source_position
    from .normalisation import providers

    params = {
        Filename: filename,
        FrameTimeMonitor: monitor,
        MonitorPosition: monitor.coords['position'],
        SourceName: source_name,
        TimeOfFlightLookupTable: tof_lookup_table,
    }
    pipeline = Pipeline((*providers, source_position), params=params)
    return pipeline.compute(TofMonitor)


def get_normalization_monitor(monitors, monitor_component, collapse: bool = False):
    """Get the data of the named monitor component, converting frame_time to nanoseconds
    to match event_time_offset

    Parameters
    ----------
    monitors:
        A dictionary mapping monitor component name to loaded scippneutron.NXmonitor
        groups
    monitor_component:
        The name of the monitor component to access
    collapse: bool
        For some simulated experiments, a parameter was scanned which should not be
        treated as separate time points. When provided True, these points are integrated
        over the 'time' dimension.

    Returns
    -------
    :
        Monitor data with frame_time converted to nanoseconds to match the timescale
        used for events
    """
    normalization = monitors[monitor_component]
    if collapse:
        # This is very specialized to how the simulated scans are done,
        # it needs to be generalized?
        normalization = normalization.sum(dim='time')
    # rescale the frame_time axis. Why does it need to be done this way?
    return normalization.transform_coords(
        ['frame_time'], graph={'frame_time': lambda t: t.to(unit='nanosecond')}
    )


def add_energy_coordinates(sample_events, ki_params, kf_params):
    """Extract incident_energy, final_energy, and energy_transfer

    Parameters
    ----------
    sample_events:
        Events with time of flight to the sample.
    ki_params:
        A dictionary of parameters needed by the incident-spectrometer sciline pipeline
    kf_params:
        A dictionary of parameters needed by the secondary-spectrometer sciline pipeline

    Returns
    -------
    :
        ``sample_events`` with added 'incident_energy', 'final_energy', and
        'energy_transfer' coordinates.
    """
    from sciline import Pipeline
    from scippneutron.conversion.tof import energy_from_tof

    from ..types import (
        EnergyTransfer,
        FinalEnergy,
        FinalWavenumber,
        IncidentEnergy,
    )
    from .conservation import providers

    params = {}
    params.update(ki_params)
    params.update(kf_params)
    pipeline = Pipeline(providers, params=params)
    pipeline[FinalWavenumber] = pipeline.get(FinalWavenumber).compute()

    incident_graph = {
        'incident_energy': lambda sample_tof, L1: energy_from_tof(
            tof=sample_tof, Ltotal=L1
        ),
        # TODO remove once L1 is precomputed
        'L1': lambda: pipeline.compute(SourceSamplePathLength),
    }
    events_with_energy_axes = sample_events.transform_coords(
        'incident_energy',
        graph=incident_graph,
        keep_intermediate=False,
    )
    pipeline[IncidentEnergy] = events_with_energy_axes.bins.coords['incident_energy']
    energies = pipeline.compute((EnergyTransfer, FinalEnergy))
    events_with_energy_axes.bins.coords['energy_transfer'] = energies[
        EnergyTransfer
    ].to(unit='meV')
    events_with_energy_axes.coords['final_energy'] = energies[FinalEnergy]
    return events_with_energy_axes


def add_momentum_coordinates(ki_params, kf_params, events, a3: Variable):
    """Extract momentum transfer in the lab and sample-table coordinate systems

    Parameters
    ----------
    ki_params:
        A dictionary of parameters needed by the incident-spectrometer sciline pipeline
    kf_params:
        A dictionary of parameters needed by the secondary-spectrometer sciline pipeline
    events:
        The event data to which the calculated momentum components are appended
    a3:
        The scalar value of the sample rotation angle describing the events

    Returns
    -------
    :
        The event data with the two horizontal plane components of the
        momentum transfer added in the laboratory coordinate system (independent of a3)
        and the sample-table coordinate system (rotated by a3 around y).
        These new coordinates are named 'lab_momentum_x', 'lab_momentum_z',
        'table_momentum_x' and 'table_momentum_z'
    """
    from sciline import Pipeline

    from ..types import (
        FinalWavevector,
        IncidentDirection,
        SampleTableAngle,
    )
    from .conservation import (
        PARALLEL,
        PERP,
        providers,
        sample_table_momentum_vector,
    )

    if a3.size != 1:
        raise ValueError(f'Expected a3 to have 1-entry, not {a3.size}')

    params = {}
    # First we must add the lab momentum vector, since it is not a3 dependent
    params.update(ki_params)
    params.update(kf_params)
    params[SampleTableAngle] = a3

    pipeline = Pipeline(providers, params=params)
    intermediates = pipeline.compute((IncidentDirection, FinalWavevector))
    incident_direction = intermediates[IncidentDirection]
    final_wavevector = intermediates[FinalWavevector]

    def compute_lab_momentum_transfer(incident_wavelength: sc.Variable) -> sc.Variable:
        return final_wavevector - 2 * np.pi * incident_direction / incident_wavelength

    graph = {
        'lab_momentum_transfer': compute_lab_momentum_transfer,
        'lab_momentum_x': lambda lab_momentum_transfer: sc.dot(
            PERP, lab_momentum_transfer
        ),
        'lab_momentum_z': lambda lab_momentum_transfer: sc.dot(
            PARALLEL, lab_momentum_transfer
        ),
        'table_momentum_transfer': lambda lab_momentum_transfer: sample_table_momentum_vector(  # noqa: E501
            a3, lab_momentum_transfer
        ),
        'table_momentum_x': lambda table_momentum_transfer: sc.dot(
            PERP, table_momentum_transfer
        ),
        'table_momentum_z': lambda table_momentum_transfer: sc.dot(
            PARALLEL, table_momentum_transfer
        ),
    }
    return events.transform_coords(
        ('lab_momentum_x', 'lab_momentum_z', 'table_momentum_x', 'table_momentum_z'),
        graph=graph,
        keep_intermediate=False,
    )


def add_wavelength_coordinate(ki_params, events, monitor):
    """Convert to incident wavelength per event and independent monitor axis

    Parameters
    ----------
    ki_params:
        A dictionary of parameters needed by the incident-spectrometer sciline pipeline
    events:
        Event data, presumably with per-event incident energy (or wavelength, or
        inverse velocity == slowness) already calculated; the basis for the returned
        events
    monitor:
        A beam monitor with one independent axis (time since last pulse as measured).
        The monitor intensity is expected to be a histogram along the one independent
        axis, but event monitor data should work as well.

    Returns
    -------
    :
        The events with a new coordinate, 'incident_wavelength'.
        And the monitor with 'incident_wavelength' coordinate.
    """
    from sciline import Pipeline
    from scippneutron.conversion.graph import beamline
    from scippneutron.conversion.tof import wavelength_from_tof

    from .ki import providers as ki_providers

    pipeline = Pipeline(ki_providers, params=ki_params)

    incident_graph = {
        'incident_wavelength': wavelength_from_tof,
        # TODO remove once L1 is precomputed
        'L1': lambda: pipeline.compute(SourceSamplePathLength),
        'Ltotal': 'L1',  # using sample times
        'tof': 'sample_tof',
    }
    events_with_wavelength = events.transform_coords(
        'incident_wavelength',
        graph=incident_graph,
        keep_inputs=False,
        keep_intermediate=False,
        keep_aliases=False,
    )

    monitor_graph = {
        **beamline.beamline(scatter=False),
        'incident_wavelength': wavelength_from_tof,
    }
    wavelength_monitor = monitor.transform_coords(
        'incident_wavelength',
        graph=monitor_graph,
        keep_intermediate=False,
        keep_aliases=False,
    )

    return events_with_wavelength, wavelength_monitor


def get_geometric_a4(kf_params):
    from sciline import Pipeline

    from ..types import DetectorGeometricA4
    from .kf import providers

    pipeline = Pipeline(providers, params=kf_params)
    geometric_a4 = pipeline.compute(DetectorGeometricA4)
    return geometric_a4


def normalise_wavelength_events(ki_params, kf_params, events, monitor):
    from sciline import Pipeline

    from ..types import (
        WavelengthBins,
        WavelengthEvents,
        WavelengthMonitor,
    )
    from .normalisation import providers

    params = {
        WavelengthEvents: events,
        WavelengthMonitor: monitor,
        WavelengthBins: monitor.coords['incident_wavelength'],
    }
    params.update(ki_params)
    params.update(kf_params)
    pipeline = Pipeline(providers, params=params)
    events = pipeline.compute(NormWavelengthEvents)
    return events


def split(
    triplets,
    analyzers,
    monitors,
    logs,
    a3_name: str | None = None,
    a4_name: str | None = None,
):
    """Use the (a3, a4) logged value pairs to split triplet, analyzer, and monitor data
    into single-setting sets

    Parameters
    ----------
    triplets: DataArray
        The triplet positions _should_ change with a4 (if simulated, this is certainly
        not implemented correctly yet). The event data they contain depends on
        (a3, a4) [plus any other time-dependent parameter] so must be split.
    analyzers:
        The analyzer position _should_ change with a4
        (if simulated, this is certainly not implemented correctly yet)
    monitors: DataArray
        The histogram (simulated, or real current beam monitor) or event
        (real fission monitor, etc.?) data is time-dependent and used to normalize
        the detector data. It must therefore be split into (a3, a4) sets
    logs: DataGroup
        Entries built from NXlogs in the real or simulated instrument
    a3_name: str
        The name of the sample table angle log entry in `logs`, 'a3' for simulated data
    a4_name: str
        The name of the detector tank angle log entry in `logs`, 'a4' for simulated data

    Returns
    -------
    :
        A list[[triplet, analyzer, monitor]] of individual (a3, a4) setting(s)
    """
    from scipp import lookup

    from ..utils import is_in_coords

    if a3_name is None:
        a3_name = 'a3'
    if a4_name is None:
        a4_name = 'a4'

    if a3_name not in logs or a4_name not in logs:
        logger.warning('Missing a3 or a4, so split performed')
        return [[triplets, analyzers, monitors]]

    a3 = lookup(logs[a3_name], 'time')
    a4 = lookup(logs[a4_name], 'time')

    event_graph = {
        'a3': lambda event_time_zero: a3[event_time_zero],
        'a4': lambda event_time_zero: a4[event_time_zero],
    }
    histogram_graph = {'a3': lambda time: a3[time], 'a4': lambda time: a4[time]}

    def do_split(x, time_name):
        graph = event_graph if 'event_time_zero' == time_name else histogram_graph
        if is_in_coords(x, time_name):
            x = x.transform_coords(('a3', 'a4'), graph=graph)
            if x.bins is not None:
                x = x.group('a3', 'a4')
        return x

    vals = [
        do_split(x, t)
        for x, t in (
            (triplets, 'event_time_zero'),
            (analyzers, 'time'),
            (monitors, 'time'),
        )
    ]

    # FIXME this only works because v.sizes['a4'] is always 1 at the moment
    vals = [
        v.flatten(['a3', 'a4'], to='time') if 'a3' in v.dims and 'a4' in v.dims else v
        for v in vals
    ]

    n_time = [v.sizes['time'] for v in vals if 'time' in v.dims]
    if len(n_time):
        if not all(n == n_time[0] for n in n_time):
            raise ValueError("Not all values have the same 'time' dimension")
        n_time = n_time[0]
        vals = [
            [v['time', i] if 'time' in v.dims else v for v in vals]
            for i in range(n_time)
        ]
    else:
        vals = [vals]

    return vals


def _get_detector_names(filename: Filename) -> list[str]:
    with snx.File(filename) as f:
        return list(f['entry/instrument'][snx.NXdetector])


def load_everything(filename: Filename):
    """Load all needed information from the named NeXus HDF5 file

    Parameters
    ----------
    filename:
        The name of the file to load data from, must have both and 'instrument'
        and 'parameters' group under 'entry'

    Returns
    -------
    sample:
        The loaded sample component group
    triplets:
        All scippnexus.NXdetector groups under 'entry/instrument'
    analyzers:
        All scippnexus.NXcrystal groups under 'entry/instrument'
    choppers:
        All scippnexus.NXdisk_chopper groups under 'entry/instrument'
    monitors:
        All scippnexus.NXmonitor groups under 'entry/instrument'
    logs:
        The scippnexus.NXlog groups named 'a3' and 'a4' under 'entry/parameters'
    """
    from ess.bifrost.types import (
        FrameMonitor0,
        FrameMonitor1,
        FrameMonitor2,
        FrameMonitor3,
    )
    from ess.bifrost.workflow import BifrostSimulationWorkflow

    monitor_keys = (
        FrameMonitor0,
        FrameMonitor1,
        FrameMonitor2,
        FrameMonitor3,
    )
    detector_names = _get_detector_names(filename)

    workflow = BifrostSimulationWorkflow()
    # Use [SampleRun] for now until we process multiple runs together.
    workflow[Filename[SampleRun]] = filename
    workflow[PreopenNeXusFile] = PreopenNeXusFile(True)
    workflow[DetectorData[SampleRun]] = (
        workflow[DetectorData[SampleRun]]
        .map({NeXusDetectorName: detector_names})
        .reduce(func=lambda *x: x)
    )

    loaded = workflow.compute(
        [
            DetectorData[SampleRun],
            Position[snx.NXsample, SampleRun],
            Analyzers[SampleRun],
            Choppers[SampleRun],
            InstrumentAngles[SampleRun],
            *(MonitorData[SampleRun, key] for key in monitor_keys),
            *(NeXusMonitorName[key] for key in monitor_keys),
        ]
    )
    sample = sc.DataGroup(position=loaded[Position[snx.NXsample, SampleRun]])
    analyzers = loaded[Analyzers[SampleRun]]
    choppers = loaded[Choppers[SampleRun]]
    monitors = sc.DataGroup(
        {
            loaded[NeXusMonitorName[key]]: loaded[MonitorData[SampleRun, key]]
            for key in monitor_keys
        }
    )
    triplets = sc.DataGroup(
        zip(detector_names, loaded[DetectorData[SampleRun]], strict=True)
    )
    instrument_angles = loaded[InstrumentAngles[SampleRun]]

    return sample, triplets, analyzers, choppers, monitors, instrument_angles


def one_setting(
    sample,
    triplet_events,
    analyzers,
    norm_monitor,
    filename,
    names,
    tof_lookup_table,
    warn_about_a3=True,
):
    """Calculate the event properties for a single (a3, a4) setting"""
    detector_positions = triplet_events.coords['position']
    kf_params, sample_detector_flight_time = find_sample_detector_flight_time(
        sample, analyzers, detector_positions
    )
    sample_events = get_sample_events(triplet_events, sample_detector_flight_time)
    ki_params, unwrapped_sample_events, primary = get_unwrapped_sample_events(
        filename=filename,
        source_name=names['source'],
        sample_name=names['sample'],
        sample_events=sample_events,
        tof_lookup_table=tof_lookup_table,
    )

    unwrapped_norm_monitor = get_unwrapped_monitor(
        filename=filename,
        monitor=norm_monitor,
        source_name=names['source'],
        tof_lookup_table=tof_lookup_table,
    )

    unwrapped_sample_events = add_energy_coordinates(
        unwrapped_sample_events, ki_params, kf_params
    )
    unwrapped_sample_events.coords['theta'] = get_geometric_a4(kf_params)

    if 'a3' in triplet_events.coords:
        # this _should_ be one (a3, a4) setting,
        # with a single a3 value on triple_events (and norm_monitor)
        a3 = triplet_events.coords['a3']
    else:
        from scipp import scalar

        if warn_about_a3:
            logger.warning("No a3 present in setting, assuming 0 a3")
        a3 = scalar(0, unit='deg')

    # Set up the normalisation by adding an 'incident_wavelength' coordinate to
    # the individual events and the normalisation monitor
    unwrapped_sample_events, monitor = add_wavelength_coordinate(
        ki_params,
        unwrapped_sample_events,
        unwrapped_norm_monitor,
    )
    unwrapped_sample_events = add_momentum_coordinates(
        ki_params, kf_params, unwrapped_sample_events, a3
    )

    norm_events = normalise_wavelength_events(
        ki_params, kf_params, unwrapped_sample_events, monitor
    )

    return {
        'triplet_events': triplet_events,
        'events': unwrapped_sample_events,
        'norm_monitor': norm_monitor,
        'sample_detector_flight_time': sample_detector_flight_time,
        'analyzers': analyzers,
        'wavelength_monitor': monitor,
        'norm_events': norm_events,
    }


def load_precompute(
    filename: Filename,
    named_components: dict[str, str],
    is_simulated: bool = False,
):
    """Load data from a NeXus file and perform (a3, a4) independent calculations

    Parameters
    ----------
    filename:
        The file which contains the data to load
    named_components:
        The file-specific names of (at least) the source, sample and normalization
        monitor group names under 'entry/instrument'
    is_simulated:
        A flag to indicate if the file comes from a McStas simulation, such that
        the event_time_offset needs to be modified to look like real data.

    Returns
    -------
    sample:
        The sample group loaded from the NeXus file
    analyzers:
        A single array with all analyzer information needed to calculate
        secondary-spectrometer parameters
    triplet_events:
        A single array with the events and detector-pixel information needed
        to calculate parameters
    norm_monitor:
        The normalization monitor data with frame_time converted to nanoseconds
    logs:
        A dictionary of the 'a3' and 'a4' logs from the 'entry/parameter' group
        in the NeXus file
    """
    sample, triplets, analyzers, choppers, monitors, logs = load_everything(filename)

    if is_simulated:
        for name in triplets:
            triplets[name] = convert_simulated_time_to_event_time_offset(triplets[name])

    analyzers = combine_analyzers(analyzers, triplets)
    # detectors = combine_detectors(triplets)
    triplet_events = get_triplet_events(triplets.values())

    norm_monitor = get_normalization_monitor(monitors, named_components['monitor'])
    return sample, analyzers, triplet_events, norm_monitor, logs


def component_names(
    source_component: str | None = None,
    sample_component: str | None = None,
    monitor_component: str | None = None,
    is_simulated: bool = False,
):
    """Return a dictionary mapping component type to component name

    Parameters
    ----------
    source_component: str
        The user-provided source component name, should exist at
        'entry/instrument/{source_component}' in the datafile
    sample_component: str
        The user-provided sample component name, should exist at
        'entry/instrument/{sample_component}' in the datafile
    monitor_component: str
        The user-provided normalization monitor component name, should exist at
        'entry/instrument/{monitor_component}'
    is_simulated: bool
        If true, user-provided names will be augmented with the McStas component
        names for the specific types.

    Returns
    -------
    :
        A dictionary mapping component type name to group name
    """
    names = {
        'source': source_component,
        'sample': sample_component,
        'monitor': monitor_component,
    }
    if is_simulated:
        sim_components = {
            'source': '001_ESS_source',
            'sample': '114_sample_stack',
            'monitor': '110_frame_3',
        }
        for k, v in sim_components.items():
            if names[k] is None:
                names[k] = v
    return names


def bifrost(
    filename: Filename,
    tof_lookup_table: TimeOfFlightLookupTable,
    source_component: str | None = None,
    sample_component: str | None = None,
    monitor_component: str | None = None,
    is_simulated: bool = False,
):
    """Load a BIFROST data file and convert to S(Q,E)
    in the sample-table coordinate system

    Parameters
    ----------
    filename:
        The name of the NeXus file to load
    tof_lookup_table:
        Time-of-flight lookup table as produced by
        `ESSreduce <https://scipp.github.io/essreduce/user-guide/tof/frame-unwrapping.html>`_.
    source_component:
        The group name under 'entry/instrument' in the NeXus file containing
        source information
    sample_component:
        The group name under 'entry/instrument' in the NeXus file containing
        sample information
    monitor_component:
        The group name under 'entry/instrument' in the NeXus file containing
        normalization monitor information
    is_simulated:
        Whether the NeXus file comes from a McStas simulation, in which case default
        component names are set if not provided and the data is modified to
        look like real data

    Returns
    -------
    A dictionary of data from the workflow, concatenated along a 'setting' dimension
    corresponding to separate (a3, a4) grouped data. The entries in the dictionary
    may not all be useful, and are subject to pruning as experience is gained with
    the workflow.
    """
    from scipp import concat
    from tqdm import tqdm

    named_components = component_names(
        source_component,
        sample_component,
        monitor_component,
        is_simulated,
    )
    sample, analyzers, triplet_events, norm_monitor, logs = load_precompute(
        filename, named_components, is_simulated
    )
    settings = split(triplet_events, analyzers, norm_monitor, logs)
    data = [
        one_setting(
            sample,
            one_triplet_events,
            one_analyzers,
            one_monitor,
            filename,
            named_components,
            tof_lookup_table=tof_lookup_table,
        )
        for one_triplet_events, one_analyzers, one_monitor in tqdm(
            settings, desc='(a3, a4) settings'
        )
    ]
    return {k: concat([d[k] for d in data], 'setting') for k in data[0]}


def bifrost_single(
    filename: Filename,
    tof_lookup_table: TimeOfFlightLookupTable,
    source_component: str | None = None,
    sample_component: str | None = None,
    monitor_component: str | None = None,
    is_simulated: bool = False,
    extras: bool = False,
):
    """Load a BIFROST data file and convert to S(Q,E)
    in the laboratory coordinate system

    Parameters
    ----------
    filename:
        The name of the NeXus file to load
    tof_lookup_table:
        Time-of-flight lookup table as produced by
        `ESSreduce <https://scipp.github.io/essreduce/user-guide/tof/frame-unwrapping.html>`_.
    source_component:
        The group name under 'entry/instrument' in the NeXus file containing
        source information
    sample_component:
        The group name under 'entry/instrument' in the NeXus file containing
        sample information
    monitor_component:
        The group name under 'entry/instrument' in the NeXus file containing
        normalization monitor information
    is_simulated:
        Whether the NeXus file comes from a McStas simulation, in which case default
        component names are set if not provided and the data is modified to look like
        real data
    extras:
        If true, the loaded sample group and 'a3' and 'a4' logs will be returned in
        the dictionary

    Returns
    -------
    A dictionary of data from the workflow. The entries in the dictionary may not
    all be useful, and are subject to pruning as experience is gained with the workflow.
    """
    named_components = component_names(
        source_component,
        sample_component,
        monitor_component,
        is_simulated,
    )
    sample, analyzers, triplet_events, norm_monitor, logs = load_precompute(
        filename, named_components, is_simulated
    )
    if 'time' in norm_monitor.sizes:
        norm_monitor = norm_monitor.sum('time')

    data = one_setting(
        sample,
        triplet_events,
        analyzers,
        norm_monitor,
        filename,
        named_components,
        tof_lookup_table=tof_lookup_table,
        warn_about_a3=False,
    )

    if extras:
        data['sample'] = sample
        data['logs'] = logs

    return data


def bifrost_to_nxspe(
    *,
    output: NXspeFileName,
    filename: Filename[SampleRun] | None = None,
    events: NormWavelengthEvents | None = None,
    **kwargs,
):
    from sciline import Pipeline

    from ..types import NXspeFileNames
    from .io import providers as io_providers

    if filename is None and events is None:
        raise ValueError("Provide events, or filename to read and reduce file")
    if events is None:
        reduced = bifrost(filename, **kwargs)
        events = reduced['norm_events']

    pipeline = Pipeline(
        providers=io_providers,
        params={
            NXspeFileName: output,
            NormWavelengthEvents: events,
        },
    )
    return pipeline.compute(NXspeFileNames)
