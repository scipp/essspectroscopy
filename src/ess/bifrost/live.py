# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Live data reduction workflows for BIFROST."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NewType

import sciline
import scipp as sc

from ess.spectroscopy.types import (
    EnergyData,
    NeXusDetectorName,
    RunType,
)

from .workflow import BifrostWorkflow


@dataclass(frozen=True, kw_only=True, slots=True)
class CutAxis:
    """Axis and bins for cutting 4D Q - delta E data.

    Each axis defines a projection of the :math:`Q` - :math:`\\Delta E` space
    onto a 1D line as well as bin edges on that line.

    Examples
    --------
    Cut along :math:`Q_x`: (see also :meth:`CutAxis.from_q_vector`)

        >>> from ess.bifrost.live import CutAxis
        >>> axis = CutAxis(
        ...     output='Qx',
        ...     fn=lambda sample_table_momentum_transfer: sc.dot(
        ...         sc.vector([1, 0, 0]),
        ...         sample_table_momentum_transfer,
        ...     ),
        ...     bins=sc.linspace(dim='Qx', start=-0.5, stop=0.5, num=100, unit='1/Å'),
        ... )

    Cut along the norm :math:`|Q|`:
    (Note that ``sc.norm`` is wrapped in a lambda to use the
    proper name for the input coordinate, see :attr:`CutAxis.fn`.)

        >>> axis = CutAxis(
        ...     output='|Q|',
        ...     fn=lambda sample_table_momentum_transfer: sc.norm(
        ...         sample_table_momentum_transfer
        ...     ),
        ...     bins=sc.linspace(dim='|Q|', start=-0.9, stop=3.0, num=100, unit='1/Å'),
        ... )

    Cut along :math:`\\Delta E`:

        >>> axis = CutAxis(
        ...     output='E',
        ...     fn=lambda energy_transfer: energy_transfer,
        ...     bins=sc.linspace('E', -0.1, 0.1, 300, unit='meV')
        ... )
    """

    output: str
    """Name of the output coordinate."""
    fn: Callable[[...], sc.Variable]
    """Function to perform the cut.

    Used in :func:`scipp.transform_coords` and so should request input coordinates
    by name.
    """
    bins: sc.Variable
    """Bin edges for the cut."""

    @classmethod
    def from_q_vector(cls, output: str, vec: sc.Variable, bins: sc.Variable):
        """Construct from an arbitrary direction in Q."""
        vec = vec / sc.norm(vec)
        return cls(
            output=output,
            fn=lambda sample_table_momentum_transfer: sc.dot(
                vec, sample_table_momentum_transfer
            ),
            bins=bins,
        )


CutAxis1 = NewType('CutAxis1', CutAxis)
"""Sciline domain type for cut axis 1."""
CutAxis2 = NewType('CutAxis2', CutAxis)
"""Sciline domain type for cut axis 1."""


class CutData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data that was cut along CutAxis1 and CutAxis2."""


def cut(
    data: EnergyData[RunType], *, axis_1: CutAxis1, axis_2: CutAxis2
) -> CutData[RunType]:
    """Cut data along two axes.

    This function projects the input ``data`` expressed in :math:`Q` and
    :math:`\\Delta E` onto a 2D surface defined by the cut axes.
    This integrates over the other dimensions.
    Then, the projected data is histogrammed according to the axis bins.

    Parameters
    ----------
    data:
        Input data with coordinates "sample_table_momentum_transfer" and
        "energy_transfer".
    axis_1:
        Defines the projection onto and binning in the first axis.
    axis_2:
        Defines the projection onto and binning in the second axis.

    Returns
    -------
    :
        ``data`` projected and histogrammed along the cut axes.
    """
    new_coords = {axis_1.output, axis_2.output}
    projected = data.bins.concat().transform_coords(
        new_coords,
        graph={axis_1.output: axis_1.fn, axis_2.output: axis_2.fn},
        keep_inputs=False,
    )
    projected = projected.drop_coords(list(set(projected.coords.keys()) - new_coords))
    return CutData[RunType](
        projected.hist({axis_2.output: axis_2.bins, axis_1.output: axis_1.bins})
    )


def BifrostQCutWorkflow(detector_names: list[NeXusDetectorName]) -> sciline.Pipeline:
    """Workflow for BIFROST to compute cuts in Q-E-space."""
    workflow = BifrostWorkflow(detector_names)
    workflow.insert(cut)
    return workflow
