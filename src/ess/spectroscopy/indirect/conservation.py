# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import scipp as sc
from scippneutron.conversion.beamline import BeamAlignedUnitVectors

from ..types import (
    EnergyTransfer,
    FinalEnergy,
    FinalWavenumber,
    FinalWavevector,
    GravityVector,
    IncidentEnergy,
    IncidentWavenumber,
    IncidentWavevector,
    LabMomentumTransfer,
    LabMomentumTransferX,
    LabMomentumTransferY,
    LabMomentumTransferZ,
    SampleTableAngle,
    TableMomentumTransfer,
    TableMomentumTransferX,
    TableMomentumTransferY,
    TableMomentumTransferZ,
)
from .kf import providers as kf_providers
from .ki import providers as ki_providers


# TODO should be in bifrost module
def beam_aligned_unit_vectors(gravity: GravityVector) -> BeamAlignedUnitVectors:
    from scippneutron.conversion.beamline import beam_aligned_unit_vectors

    return beam_aligned_unit_vectors(
        gravity=gravity,
        # This is not the same as `sample_position-source_position`
        incident_beam=sc.vector([0, 0, 1], unit='m'),
    )


def lab_momentum_vector(
    incident_wavevector: IncidentWavevector, final_wavevector: FinalWavevector
) -> LabMomentumTransfer:
    """Return the momentum transferred to the sample in the laboratory coordinate system

    The laboratory coordinate system is independent of sample angle

    Parameters
    ----------
    incident_wavevector:
        incident wavevector of the neutron
    final_wavevector:
        final wavevector of the neutron

    Returns
    -------
    :
        The difference kf - ki
    """
    return final_wavevector - incident_wavevector


def lab_momentum_x(
    lab_momentum_vec: LabMomentumTransfer,
    beam_aligned_unit_x: sc.Variable,
) -> LabMomentumTransferX:
    """Return the X coordinate of the momentum transfer in the lab coordinate system"""
    return sc.dot(beam_aligned_unit_x, lab_momentum_vec)


def lab_momentum_y(
    lab_momentum_vec: LabMomentumTransfer,
    beam_aligned_unit_y: sc.Variable,
) -> LabMomentumTransferY:
    """Return the Y coordinate of the momentum transfer in the lab coordinate system"""
    return sc.dot(beam_aligned_unit_y, lab_momentum_vec)


def lab_momentum_z(
    lab_momentum_vec: LabMomentumTransfer,
    beam_aligned_unit_z: sc.Variable,
) -> LabMomentumTransferZ:
    """Return the Z coordinate of the momentum transfer in the lab coordinate system"""
    return sc.dot(beam_aligned_unit_z, lab_momentum_vec)


def sample_table_momentum_vector(
    a3: SampleTableAngle,
    lab_momentum_vec: LabMomentumTransfer,
    beam_aligned_unit_y: sc.Variable,
) -> TableMomentumTransfer:
    """Rotate the momentum transfer vector into the sample-table coordinate system

    Notes
    -----
    When a3 is zero, the sample-table and lab coordinate systems are the same.
    That is, Z is along the incident beam, Y is opposite the gravitational force,
    and X completes the right-handed coordinate system. The sample-table angle, a3,
    has a rotation vector along Y, such that a positive 90-degree rotation places the
    sample-table Z along the lab X.

    Parameters
    ----------
    a3:
        The rotation angle of the sample table around the laboratory Y axis
    lab_momentum_vec:
        The momentum transfer in the laboratory coordinate system
    """
    # negative a3 since we rotate coordinates not axes here
    y = beam_aligned_unit_y
    return sc.spatial.rotations_from_rotvecs(-a3 * y) * lab_momentum_vec


def sample_table_momentum_x(
    table_momentum_vec: TableMomentumTransfer, beam_aligned_unit_x: sc.Variable
) -> TableMomentumTransferX:
    """Return the X coordinate of the momentum transfer in the sample-table system"""
    return sc.dot(beam_aligned_unit_x, table_momentum_vec)


def sample_table_momentum_y(
    table_momentum_vec: TableMomentumTransfer,
    beam_aligned_unit_y: sc.Variable,
) -> TableMomentumTransferY:
    """Return the Y coordinate of the momentum transfer in the sample-table system"""
    return sc.dot(beam_aligned_unit_y, table_momentum_vec)


def sample_table_momentum_z(
    table_momentum_vec: TableMomentumTransfer,
    beam_aligned_unit_z: sc.Variable,
) -> TableMomentumTransferZ:
    """Return the Z coordinate of the momentum transfer in the sample-table system"""
    return sc.dot(beam_aligned_unit_z, table_momentum_vec)


def energy(
    incident_wavenumber: IncidentWavenumber, final_wavenumber: FinalWavenumber
) -> EnergyTransfer:
    """Calculate the energy transferred to the sample by a neutron"""
    from scipp.constants import hbar, neutron_mass

    return hbar**2 / 2 / neutron_mass * (incident_wavenumber**2 - final_wavenumber**2)


def energy_transfer(
    incident_energy: IncidentEnergy, final_energy: FinalEnergy
) -> EnergyTransfer:
    return incident_energy - final_energy


def graph():
    from .kf import graph as kf_graph
    from .ki import graph as ki_graph

    # depends on ki, kf, a3, gravity
    return {
        **ki_graph(),
        **kf_graph(),
        (
            'beam_aligned_unit_x',
            'beam_aligned_unit_y',
            'beam_aligned_unit_z',
        ): beam_aligned_unit_vectors,
        'lab_momentum_vec': lab_momentum_vector,
        'lab_momentum_x': lab_momentum_x,
        'lab_momentum_y': lab_momentum_y,
        'lab_momentum_z': lab_momentum_z,
        'table_momentum_vec': sample_table_momentum_vector,
        'table_momentum_x': sample_table_momentum_x,
        'table_momentum_y': sample_table_momentum_y,
        'table_momentum_z': sample_table_momentum_z,
        'energy_transfer': energy,
    }


providers = (
    *ki_providers,
    *kf_providers,
    lab_momentum_vector,
    lab_momentum_x,
    lab_momentum_y,
    lab_momentum_z,
    sample_table_momentum_vector,
    sample_table_momentum_x,
    sample_table_momentum_y,
    sample_table_momentum_z,
    energy_transfer,
)
