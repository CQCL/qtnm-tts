"""Tests for the LCU circuit implementation."""

import pytest
import numpy as np
from qtnmtts.circuits.utils import (
    block_encoded_sparse_matrix,
)
from qtnmtts.measurement.utils import circuit_unitary_postselect
from qtnmtts.circuits.lcu import LCUMultiplexorBox
from pytket.utils import QubitPauliOperator
from pytest_lazyfixture import lazy_fixture
from qtnmtts.circuits.utils import is_hermitian
from qtnmtts.circuits.utils._testing import qcontrol_test


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_fixture"),
    ],
)
@pytest.mark.parametrize("LCUBox", [LCUMultiplexorBox])
def test_lcu_op(LCUBox: type, op: QubitPauliOperator):
    """Test that operator obtained from postselecting the LCU.

    Circuit Must be the same as the block encoded operator itself

    Args:
    ----
        LCUBox (LCUBox): The LCUBox to test.
        op (QubitPauliOperator): The operator to test.

    """
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    lcu_box = LCUBox(op, n_state_qubits)
    circ_h = circuit_unitary_postselect(lcu_box.get_circuit(), lcu_box.postselect)
    scipy_h = block_encoded_sparse_matrix(lcu_box).todense()
    np.testing.assert_allclose(scipy_h, circ_h, atol=1e-10)


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_hermitian_fixture"),
    ],
)
@pytest.mark.parametrize("LCUBox", [LCUMultiplexorBox])
def test_is_hermitian(LCUBox: type, op: QubitPauliOperator):
    """Test that operator obtained from postselecting the LCU.

    Circuit Must be the same as the block encoded operator itself

    Args:
    ----
        LCUBox (LCUBox): The LCUBox to test.
        op (QubitPauliOperator): The operator to test.

    """
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    lcu_box = LCUBox(op, n_state_qubits)
    assert is_hermitian(lcu_box)
    assert lcu_box.select_box.is_hermitian


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_nonhermitian_fixture"),
    ],
)
@pytest.mark.parametrize("LCUBox", [LCUMultiplexorBox])
def test_is_not_hermitian(LCUBox: type, op: QubitPauliOperator):
    """Test that operator obtained from postselecting the LCU.

    Circuit Must be the same as the block encoded operator itself

    Args:
    ----
        LCUBox (LCUBox): The LCUBox to test.
        op (QubitPauliOperator): The operator to test.

    """
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    lcu_box = LCUBox(op, n_state_qubits)
    assert not is_hermitian(lcu_box)
    assert not lcu_box.select_box.is_hermitian


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_fixture"),
    ],
)
@pytest.mark.parametrize("LCUBox", [LCUMultiplexorBox])
def test_lcu_op_qcontrol(LCUBox: type, op: QubitPauliOperator):
    """Test that operator obtained from postselecting the LCU.

    Circuit Must be the same as the block encoded operator itself

    Args:
    ----
        LCUBox (LCUBox): The LCUBox to test.
        op (QubitPauliOperator): The operator to test.

    """
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    lcu_box = LCUBox(op, n_state_qubits)
    atol = 1e-10
    qcontrol_test(lcu_box, atol)
