"""Tests for the LCU circuit implementation."""

import pytest
import numpy as np
from qtnmtts.circuits.lcu import LCUMultiplexorBox
from pytket.utils.operators import QubitPauliOperator
from qtnmtts.circuits.core import PowerBox
from numpy.linalg import matrix_power
from pytest_lazyfixture import lazy_fixture
from qtnmtts.circuits.utils._testing import qcontrol_test


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_fixture"),
    ],
)
@pytest.mark.parametrize("power", [1, 2, 3, 4])
def test_power_box(op: QubitPauliOperator, power: int):
    """Test the PowerBox.

    Test the unitaries of the PowerBox compared to the unitary
    multiplied by the power.

    Args:
    ----
        op (QubitPauliOperator): QubitPauliOperator
        power (int): power

    """
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    lcu_box = LCUMultiplexorBox(op, n_state_qubits)
    scipy_u = lcu_box.get_unitary()
    scipy_u_power = matrix_power(scipy_u, power)
    power_box = PowerBox(lcu_box, power)
    circ_u_power = power_box.get_unitary()
    np.testing.assert_allclose(scipy_u_power, circ_u_power, atol=1e-10)


def test_power_box_qcontrol(op_fixture: QubitPauliOperator):
    """Test the PowerBox.qcontrol() method."""
    n_state_qubits = (
        max(
            [
                p.index[0]
                for p_list in list(op_fixture._dict.keys())
                for p in p_list.map.keys()
            ]
        )
        + 1
    )
    lcu_box = LCUMultiplexorBox(op_fixture, n_state_qubits)
    qcontrol_box = lcu_box.power(1)
    qcontrol_test(qcontrol_box, atol=1e-10)


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_fixture"),
    ],
)
@pytest.mark.parametrize("power", [1, 2, 3, 4])
def test_power_box_dagger_control(op: QubitPauliOperator, power: int):
    """Test the PowerBox.dagger interaction with qcontrol.

    Test the unitaries of the controlled PowerBox when applying the dagger
    before and after taking the power.

    Args:
    ----
        op (QubitPauliOperator): QubitPauliOperator
        power (int): power

    """
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    lcu_box = LCUMultiplexorBox(op, n_state_qubits)
    power_dagger_box_qc = lcu_box.power(power).dagger.qcontrol(1)
    dagger_power_box_qc = lcu_box.dagger.power(power).qcontrol(1)

    np.testing.assert_allclose(
        power_dagger_box_qc.get_unitary(), dagger_power_box_qc.get_unitary(), atol=1e-10
    )
