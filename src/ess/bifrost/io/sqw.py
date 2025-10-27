# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""SQW output for BIFROST."""

import dataclasses

import numpy as np
import numpy.typing as npt
import scipp as sc
from scippneutron.io import sqw
from scippneutron.metadata import ESS_SOURCE, Beamline, Measurement

from ess.spectroscopy.types import (
    EnergyBins,
    IncidentEnergyDetector,
    OutFilename,
    PulsePeriod,
    SampleRun,
    SQWBinSizes,
)

# Projection axes
# For now, we always project onto the cartesian axes for simplicity.
_AXIS_U = sc.vector([1, 0, 0], unit="1/angstrom")
_AXIS_V = sc.vector([0, 1, 0], unit="1/angstrom")
_AXIS_W = sc.vector([0, 0, 1], unit="1/angstrom")


def save_sqw(
    path: OutFilename,
    events: IncidentEnergyDetector[SampleRun],
    *,
    bin_sizes: SQWBinSizes,
    energy_bins: EnergyBins,  # TODO distinct from above!
    beamline: Beamline,
    measurement: Measurement,
    pulse_period: PulsePeriod,
    sample: sqw.SqwIXSample,
) -> None:
    """Save events recorded at BIFROST to an SQW file.

    ``events`` has shape (*logical, a3, a4), is binned with Ei event coord.
    """
    # TODO validate inputs: single a4

    flat_events = _flatten_events(events)
    del events  # 'move' events into _flatten_events
    _filter_and_convert_coords_in_place(flat_events)

    sample_angle = flat_events.coords['a3']

    observations = _histogram_detector_setting_ei(flat_events, energy_bins=energy_bins)
    del flat_events  # 'move' flat_events into _histogram_detector_setting_ei
    final_energy = observations.coords['final_energy']
    observations = _with_inelastic_coords(observations)
    energy_transfer = observations.coords['energy_transfer'].rename_dims(
        incident_energy='energy_transfer'
    )
    observations = observations.flatten(to="observation")

    dnd, counts, binned_observations = _bin_image(observations, bin_sizes)
    del observations  # 'move' observations into _bin_image
    pix = _make_pixel_data(binned_observations)
    del binned_observations  # 'move' binned_observations into _make_pixel_data
    pix_buffer = _make_pixel_buffer(pix)
    del pix  # 'move' pix into _make_pixel_buffer

    experiments = _make_experiments(
        energy_transfer=energy_transfer,
        final_energy=final_energy,
        sample_angle=sample_angle,
    )
    builder = (
        sqw.Sqw.build(path, title=measurement.title)
        .add_default_instrument(_make_instrument(beamline, pulse_period))
        .add_default_sample(sample)
        .add_dnd_data(_make_dnd_metadata(dnd, sample), data=dnd.data, counts=counts)
        .add_pixel_data(pix_buffer, experiments=experiments)
    )
    builder.create()


def _flatten_events(events: IncidentEnergyDetector[SampleRun]) -> sc.DataArray:
    """Flatten events from (*logical, a3, a4) to (detector, setting).

    Also assigns the 'irun' coordinate as that requires the initial shape.
    """
    logical_dims = [dim for dim in events.dims if dim not in ("a3", "a4")]
    aux = events.flatten(logical_dims, 'detector')

    n_a3 = aux.sizes['a3']
    aux.coords['i_a3'] = sc.arange('a3', n_a3, dtype='float32', unit=None)
    aux.coords['i_a4'] = sc.arange('a4', aux.sizes['a4'], dtype='float32', unit=None)
    flat = aux.flatten(['a3', 'a4'], 'setting')
    return flat.assign_coords(
        irun=flat.coords.pop('i_a3')
        + flat.coords.pop('i_a4') * sc.index(n_a3)
        + sc.index(1, dtype='float32')  # 1-based indexing
    )


def _filter_and_convert_coords_in_place(flat_events: sc.DataArray) -> None:
    """Filter out all coords we no longer need and convert to float32.

    This function mainly serves to reduce memory usage.
    The conversion to float32 is ultimately required when building the 'pixel data',
    we do it here to reduce size early. This should not have a significant impact
    on precision as we do not reduce these coordinates.
    The event weights remain in float64 or int64 for now because
    they do need to be summed.
    """
    flat_events.coords['final_energy'] = flat_events.coords['final_energy'].to(
        dtype='float32', copy=False
    )
    flat_events.bins.coords['incident_energy'] = flat_events.bins.coords[
        'incident_energy'
    ].to(dtype='float32', copy=False)

    flat_events.coords['idet'] = flat_events.coords.pop('detector_number').to(
        dtype='float32', copy=False
    )
    # to 1-based indexing:
    flat_events.coords['idet'] += sc.scalar(1.0, dtype='float32', unit=None)

    keep = {
        'a3',
        'final_energy',
        'final_wavevector',
        'gravity',
        'idet',
        'incident_beam',
        'irun',
    }
    for dim in set(flat_events.coords) - keep:
        del flat_events.coords[dim]


def _histogram_detector_setting_ei(
    flat_events: sc.DataArray, *, energy_bins: EnergyBins
) -> sc.DataArray:
    """Histogram the events in (detector, setting, incident_energy).

    This also converts the weights to float32 as no further reductions
    will be performed for the 'pixel data'.
    We accept a loss of precision for the 'image data' here which will be
    converted back to float64 later.
    """
    hist = flat_events.hist(incident_energy=energy_bins)
    hist.coords['incident_energy'] = sc.midpoints(hist.coords['incident_energy']).to(
        dtype='float32', copy=False
    )
    hist.coords['ien'] = sc.arange(
        'incident_energy',
        1,  # 1-based indexing
        hist.sizes['incident_energy'] + 1,
        dtype='float32',
        unit=None,
    )

    hist.data = hist.data.to(dtype='float32', copy=False)

    return hist


def _with_inelastic_coords(observations: sc.DataArray) -> sc.DataArray:
    """Compute and assign Qx, Qy, Qz, and energy_transfer.

    This also drops all coordinates that are no longer needed.
    """
    from ess.spectroscopy.indirect.conversion import (
        energy_transfer,
        lab_momentum_transfer_from_incident_energy,
        rotate_to_sample_table_momentum_transfer,
    )

    graph = {
        'lab_momentum_transfer': lab_momentum_transfer_from_incident_energy,
        'sample_table_momentum_transfer': rotate_to_sample_table_momentum_transfer,
        'energy_transfer': energy_transfer,
    }
    aux = observations.transform_coords(
        ['sample_table_momentum_transfer', 'energy_transfer'],
        graph=graph,
        keep_inputs=False,
        keep_intermediate=False,
        quiet=True,
    )
    q = aux.coords.pop('sample_table_momentum_transfer')

    return aux.assign_coords(
        energy_transfer=aux.coords.pop('energy_transfer').to(
            dtype='float32', copy=False
        ),
        **{
            f'Q{dim}': _project_onto(q, axis).to(dtype='float32')
            for dim, axis in zip('xyz', (_AXIS_U, _AXIS_V, _AXIS_W), strict=True)
        },
    )


def _project_onto(vec: sc.Variable, axis: sc.Variable) -> sc.Variable:
    axis = axis / sc.norm(axis)
    # Optimization to avoid huge dot-products:
    match list(axis.values):
        case [1, 0, 0]:
            return vec.fields.x
        case [0, 1, 0]:
            return vec.fields.y
        case [0, 0, 1]:
            return vec.fields.z
        case _:
            return sc.dot(axis, vec)


def _bin_image(
    observations: sc.DataArray, bin_sizes: SQWBinSizes
) -> tuple[sc.DataArray, sc.Variable, sc.DataArray]:
    # Dim order to match pixel data (note the transpose for dnd)
    # TODO correct? Might not be!!
    binned = observations.bin(
        {dim: bin_sizes[dim] for dim in ('energy_transfer', 'Qz', 'Qy', 'Qx')}
    ).rename(energy_transfer='u4', Qz='u3', Qy='u2', Qx='u1')

    image = binned.transpose(['u1', 'u2', 'u3', 'u4'])
    dnd = image.bins.sum().to(dtype='float64', copy=False)
    counts = image.bins.size()

    return dnd, counts, binned


def _make_pixel_data(binned_observations: sc.DataArray) -> sc.DataArray:
    pix = binned_observations.data.bins.concat().value
    del binned_observations

    # Convert to expected units:
    return sc.DataArray(
        pix.data,
        coords={
            'u1': pix.coords['u1'].to(unit='1/Å', copy=False),
            'u2': pix.coords['u2'].to(unit='1/Å', copy=False),
            'u3': pix.coords['u3'].to(unit='1/Å', copy=False),
            'u4': pix.coords['u4'].to(unit='meV', copy=False),
            'idet': pix.coords['idet'],
            'irun': pix.coords['irun'],
            'ien': pix.coords['ien'],
        },
    )


def _make_pixel_buffer(pix: sc.DataArray) -> npt.NDArray[np.float32]:
    return np.c_[
        *(
            pix.coords[name].values
            for name in ('u1', 'u2', 'u3', 'u4', 'irun', 'idet', 'ien')
        ),
        pix.values,
        sc.stddevs(pix).values,
    ]


def _make_instrument(
    beamline: Beamline, pulse_period: PulsePeriod
) -> sqw.SqwIXNullInstrument:
    return sqw.SqwIXNullInstrument(
        name=beamline.name,
        source=sqw.SqwIXSource(
            name="ESS",
            target_name=ESS_SOURCE.name,
            frequency=1 / pulse_period,
        ),
    )


def _make_dnd_metadata(
    dnd: sc.DataArray, sample: sqw.SqwIXSample
) -> sqw.SqwDndMetadata:
    """Create the DND metadata for an SQW file.

    Parameters
    ----------
    dnd:
        DND 'image data'.
        Must be histogrammed along the Qx, Qy, Qz, and energy_transfer axes
        with linspace coordinates.
    sample:
        Sample metadata.
    """
    img_range = [
        sc.concat([dnd.coords[name].min(), dnd.coords[name].max()], dim=name)
        for name in ("u1", "u2", "u3", "u4")
    ]

    n_bins_all_dims = sc.array(
        dims=["axis"],
        values=[dnd.sizes[name] for name in ("u1", "u2", "u3", "u4")],
        unit=None,
    )

    return sqw.SqwDndMetadata(
        axes=sqw.SqwLineAxes(
            title="Instrument Axes",
            label=["Qx", "Qy", "Qz", "Delta E"],
            img_scales=[
                sc.scalar(1.0, unit="1/angstrom"),
                sc.scalar(1.0, unit="1/angstrom"),
                sc.scalar(1.0, unit="1/angstrom"),
                sc.scalar(1.0, unit="meV"),
            ],
            img_range=img_range,
            n_bins_all_dims=n_bins_all_dims,
            single_bin_defines_iax=sc.array(dims=["axis"], values=[True] * 4),
            dax=sc.arange("axis", 4, unit=None),
            offset=[
                sc.scalar(0.0, unit="1/angstrom"),
                sc.scalar(0.0, unit="1/angstrom"),
                sc.scalar(0.0, unit="1/angstrom"),
                sc.scalar(0.0, unit="meV"),
            ],
            changes_aspect_ratio=True,
        ),
        proj=sqw.SqwLineProj(
            title="Null Projection",
            lattice_spacing=sample.lattice_spacing,
            lattice_angle=sample.lattice_angle,
            offset=[
                sc.scalar(0.0, unit="1/angstrom"),
                sc.scalar(0.0, unit="1/angstrom"),
                sc.scalar(0.0, unit="1/angstrom"),
                sc.scalar(0.0, unit="meV"),
            ],
            label=["Qx", "Qy", "Qz", "Delta E"],
            u=_AXIS_U,
            v=_AXIS_V,
            w=_AXIS_W,
            non_orthogonal=False,
            type="aaa",
        ),
    )


def _make_experiments(
    *,
    energy_transfer: sc.Variable,
    final_energy: sc.Variable,
    sample_angle: sc.Variable,
) -> list[sqw.SqwIXExperiment]:
    experiment_template = sqw.SqwIXExperiment(
        run_id=0,
        efix=final_energy,
        emode=sqw.EnergyMode.indirect,
        en=energy_transfer,
        psi=sc.scalar(0.0, unit="rad"),
        u=_AXIS_U,
        v=_AXIS_V,
        omega=sc.scalar(0.0, unit="rad"),
        dpsi=sc.scalar(0.0, unit="rad"),
        gl=sc.scalar(0.0, unit="rad"),
        gs=sc.scalar(0.0, unit="rad"),
    )
    return [
        dataclasses.replace(experiment_template, run_id=i, psi=a3)
        for i, a3 in enumerate(sample_angle, 1)
    ]
