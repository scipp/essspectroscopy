# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Domain types for BIFROST.

This module supplements :mod:`ess.spectroscopy.types` with BIFROST-specific types.
"""

from typing import NewType

import sciline
import scipp as sc

from ess.reduce.nexus import types as reduce_t
from ess.spectroscopy.types import RunType

SampleRun = reduce_t.SampleRun
VanadiumRun = reduce_t.VanadiumRun

FrameMonitor1 = reduce_t.FrameMonitor1
FrameMonitor2 = reduce_t.FrameMonitor2
FrameMonitor3 = reduce_t.FrameMonitor3
PsdMonitor = NewType('PsdMonitor', int)


class ArcNumber(sciline.Scope[RunType, sc.Variable], sc.Variable): ...


class McStasDetectorData(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


ArcEnergy = NewType('ArcEnergy', sc.Variable)
