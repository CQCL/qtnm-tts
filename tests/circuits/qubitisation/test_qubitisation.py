"""Tests for the QubitiseBox implementation."""

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebval
from numpy.typing import NDArray
from pytest_lazyfixture import lazy_fixture
from pytket.circuit import Qubit
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils import QubitPauliOperator

from qtnmtts.circuits.core import PowerBox, QControlRegisterBox, QRegMap
from qtnmtts.circuits.lcu import LCUMultiplexorBox
from qtnmtts.circuits.qubitisation import (
    QubitiseBox,
    QControlQubitiseBox,
    QControlSquareQubitiseBox,
)
from qtnmtts.circuits.utils import (
    block_encoded_sparse_matrix,
)
from qtnmtts.measurement.utils import circuit_unitary_postselect


def chebyshev_power_matrix(
    mat: NDArray[np.complex128], power: int
) -> NDArray[np.complex128]:
    """Get the Chebyshev polynomial of a matrix."""
    coeffs = [0] * power + [1]

    e, v = np.linalg.eigh(mat)
    chebyshev_e = chebval(e, coeffs)
    cheb_scipy_h = v @ np.diag(chebyshev_e) @ v.conj().T

    return cheb_scipy_h


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_hermitian_fixture"),
    ],
)
@pytest.mark.parametrize("LCUBox", [LCUMultiplexorBox])
@pytest.mark.parametrize("power", list(range(9)))
def test_qubitisebox(LCUBox: type, op: QubitPauliOperator, power: int):
    """Test that operator obtained from postselecting the QubitiseBox.

    Args:
    ----
        LCUBox (LCUBox): The LCUBox to test.
        op (QubitPauliOperator): The operator to test.
        power (int): The power to raise the operator to.

    """
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    lcu_box = LCUBox(op, n_state_qubits)
    qubitise_box_power = QubitiseBox(lcu_box).power(power)
    circ_h = circuit_unitary_postselect(
        qubitise_box_power.reg_circuit, qubitise_box_power.register_box.postselect
    )

    scipy_h = block_encoded_sparse_matrix(lcu_box).todense()

    scipy_h = block_encoded_sparse_matrix(lcu_box).todense()

    scipy_h = chebyshev_power_matrix(scipy_h, power)

    np.testing.assert_allclose(circ_h, scipy_h, atol=1e-10)


def qcontrol_qubitise(qc_box: PowerBox) -> NDArray[np.complex128]:
    """Get scipy and circuit unitary for a QControlRegisterBox.

    For a QControlled form of a PowerBox, get the scipy and circuit unitary.
    and generate the a simple control circuit and returns the circuit
    unitary.

    Args:
    ----
        qc_box (QControlRegisterBox): The QControlRegisterBox.
        lcu_box (LCUBox): The LCUBox.

    Returns:
    -------
        NDArray[np.complex128]: The circuit unitary.

    """
    circ = qc_box.initialise_circuit()
    rotation = 0.1
    assert isinstance(qc_box.register_box, QControlRegisterBox)
    circ.Ry(rotation, qc_box.register_box.qreg.control[0])
    circ.X(qc_box.register_box.qreg.control[0])
    qreg_map = QRegMap(qc_box.q_registers, circ.q_registers)
    circ.add_registerbox(qc_box, qreg_map)
    circ.X(qc_box.register_box.qreg.control[0])
    circ.Ry(rotation, qc_box.register_box.qreg.control[0]).dagger()

    # HACK: This is a hack to get the postselect dict from the QControlRegisterBox
    post_select_dict = qc_box.postselect
    post_select_dict[qc_box.register_box.qreg.control[0]] = 0
    circ_u = circuit_unitary_postselect(circ, post_select_dict)

    return circ_u


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_hermitian_fixture"),
    ],
)
@pytest.mark.parametrize("LCUBox", [LCUMultiplexorBox])
@pytest.mark.parametrize("power", list(range(9)))
def test_qcontrol_qubitisebox(LCUBox: type, op: QubitPauliOperator, power: int):
    """Test the PytketQControlRegisterBox with an LCUBox."""
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    lcu_box = LCUBox(op, n_state_qubits)

    qc_qubitise = QubitiseBox(lcu_box).qcontrol(1).power(power)
    circ_u = qcontrol_qubitise(qc_qubitise)

    scipy_h = block_encoded_sparse_matrix(lcu_box).todense()

    scipy_h = chebyshev_power_matrix(scipy_h, power)

    rotation = 0.1
    factor = np.cos(rotation * np.pi / 2) ** 2
    scipy_u = (
        factor * scipy_h
        - (1 - factor)
        * QubitPauliOperator({QubitPauliString([Qubit(0)], [Pauli.I]): 1})
        .to_sparse_matrix(int(np.log2(scipy_h.shape[0])))
        .todense()
    )

    np.testing.assert_allclose(scipy_u, circ_u, atol=1e-10)


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_hermitian_fixture"),
    ],
)
@pytest.mark.parametrize("LCUBox", [LCUMultiplexorBox])
@pytest.mark.parametrize("power", [2, 4, 8])
def test_qcontrol_qubitisebox_unitary(LCUBox: type, op: QubitPauliOperator, power: int):
    """Unitary test for the squared controlled decomposition."""
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    lcu_box = LCUBox(op, n_state_qubits)
    # Test that controlling the reflection or the LCU is equivalent
    qc_qubitise = QubitiseBox(lcu_box)
    qc_qubitise_ref = QubitiseBox(lcu_box)
    qc_qubitise_ref.control_reflection = False

    np.testing.assert_allclose(
        qc_qubitise_ref.power(power).qcontrol(1).get_unitary(),
        qc_qubitise.power(power).qcontrol(1).get_unitary(),
        atol=1e-10,
    )
    # Test that the squared controlled decomposition is equivalent to the single
    qc_qubitise_power_single = PowerBox(qc_qubitise, power)
    np.testing.assert_allclose(
        qc_qubitise.power(power).qcontrol(1).get_unitary(),
        qc_qubitise_power_single.qcontrol(1).get_unitary(),
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_hermitian_fixture"),
    ],
)
@pytest.mark.parametrize("LCUBox", [LCUMultiplexorBox])
@pytest.mark.parametrize("power", list(range(9)))
def test_qubitisebox_correct_control(LCUBox: type, op: QubitPauliOperator, power: int):
    """Test the control of QubitiseBox.

    Given the power test that the correct control is used and that qcontrol.power is
    equivalent to power.qcontrol.

    Args:
    ----
        LCUBox (LCUBox): The LCUBox to test.
        op (QubitPauliOperator): The operator to test.
        power (int): The power to raise the operator to.

    """
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    lcu_box = LCUBox(op, n_state_qubits)
    qubitise_box = QubitiseBox(lcu_box)
    qubitise_box_power = qubitise_box.power(power)
    qcontrol_qubitise_box_power: PowerBox = qubitise_box.qcontrol(1).power(power)
    if power in [0, 1, 3, 5, 6, 7]:
        assert (
            qubitise_box_power.register_box.qcontrol.__name__
            == qubitise_box.qcontrol.__name__
        )
        assert isinstance(qcontrol_qubitise_box_power.register_box, QControlQubitiseBox)

    if power in [2, 4, 8]:
        assert (
            qubitise_box_power.register_box.qcontrol.__name__
            == qubitise_box._qcontrol_squared.__name__
        )

        assert isinstance(
            qcontrol_qubitise_box_power.register_box, QControlSquareQubitiseBox
        )
    qcontrol_qubitise_box_power_aux = qubitise_box_power.qcontrol(1)
    np.testing.assert_allclose(
        qcontrol_qubitise_box_power.get_unitary(),
        qcontrol_qubitise_box_power_aux.get_unitary(),
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_nonhermitian_fixture"),
    ],
)
@pytest.mark.parametrize("LCUBox", [LCUMultiplexorBox])
def test_qubitisebox_nonhermitian(LCUBox: type, op: QubitPauliOperator):
    """Test error on non Hermitian operators for QubitiseBox.

    Args:
    ----
        LCUBox (LCUBox): The LCUBox to test.
        op (QubitPauliOperator): The operator to test.
        power (int): The power to raise the operator to.

    """
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    lcu_box = LCUBox(op, n_state_qubits)
    with pytest.raises(
        ValueError, match="QubitiseBox only available for Hermitian LCUs."
    ):
        QubitiseBox(lcu_box)
