# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Cutting BIFROST data."""

from collections.abc import Callable

import scipp as sc

from ess.spectroscopy.types import (
    DataGroupedByRotation,
    DetectorData,
    InstrumentRotation,
    RunType,
    SampleRotation,
)


def group_by_rotation(
    data: DetectorData[RunType],
    sample_rotation: SampleRotation[RunType],
    instrument_rotation: InstrumentRotation[RunType],
) -> DataGroupedByRotation[RunType]:
    """Group data by rotation angles.

    Parameters
    ----------
    data:
        Detector events with time coordinates.
    sample_rotation:
        Sample rotation angle "a3".
        May be time-dependent (1d array with dim "time") or scalar.
    instrument_rotation:
        Instrument rotation angle "a4".
        May be time-dependent (1d array with dim "time") or scalar.

    Returns
    -------
    :
        ``data`` grouped by rotation angles "a3" and "a4".
    """
    graph = {
        'a3': _make_angle_from_time_calculator(sample_rotation),
        'a4': _make_angle_from_time_calculator(instrument_rotation),
    }
    grouped = data.transform_coords(('a3', 'a4'), graph=graph).group('a3', 'a4')
    return DataGroupedByRotation[RunType](grouped)


def _make_angle_from_time_calculator(angle: sc.DataArray) -> Callable[..., sc.Variable]:
    if angle.ndim == 0:
        return lambda: angle.data
    else:
        lut = sc.lookup(angle, 'time')
        return lambda event_time_zero: lut[event_time_zero]


providers = (group_by_rotation,)
