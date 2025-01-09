# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Neutron source information."""

from dataclasses import dataclass

import scipp as sc

from .types import SourceDuration, SourceFrequency


@dataclass(slots=True, frozen=True)
class Source:
    """Information about a neutron source."""

    pulse_duration: SourceDuration
    """The pulse duration in s."""
    frequency: SourceFrequency
    """The source frequency in Hz."""

    @property
    def period(self) -> sc.Variable:
        """The source period in ns."""
        return (1 / self.frequency).to(unit='ns')

    def to_pipeline_params(self) -> dict[type, object]:
        """Package the source parameters for a Sciline pipeline."""
        return {
            SourceDuration: self.pulse_duration,
            SourceFrequency: self.frequency,
        }


ESS_SOURCE = Source(
    frequency=sc.scalar(14.0, unit='Hz'), pulse_duration=sc.scalar(0.003, unit='s')
)
"""Default parameters of the ESS source."""
