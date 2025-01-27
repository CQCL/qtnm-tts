"""Tests for the QControlRegisterBox."""

import pytest
import numpy as np
from qtnmtts.circuits.utils import (
    block_encoded_sparse_matrix,
)
from qtnmtts.measurement.utils import circuit_unitary_postselect
from qtnmtts.circuits.lcu import LCUMultiplexorBox, LCUBox
from pytket.utils.operators import QubitPauliOperator

from qtnmtts.circuits.core import PytketQControlRegisterBox, QRegMap, QControlRegisterBox
from qtnmtts.circuits.qft import QFTBox

from pytket.pauli import Pauli, QubitPauliString

from pytket.circuit import Qubit

from pytest_lazyfixture import lazy_fixture

from numpy.typing import NDArray

from qtnmtts.circuits.utils import int_to_bits


def qcontrol_block_encode(
    qc_box: QControlRegisterBox, lcu_box: LCUBox
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Get scipy and circuit unitary for a QControlRegisterBox.

    For a QControlled form of a LCUBox, get the scipy and circuit unitary.
    and generate the a simple control circuit and return the scipy and circuit
    unitaries.

    Args:
    ----
        qc_box (QControlRegisterBox): The QControlRegisterBox.
        lcu_box (LCUBox): The LCUBox.

    """
    circ = qc_box.initialise_circuit()
    rotation = 0.1
    circ.Ry(rotation, qc_box.qreg.control[0])
    circ.X(qc_box.qreg.control[0])
    qreg_map = QRegMap(qc_box.q_registers, circ.q_registers)
    circ.add_registerbox(qc_box, qreg_map)
    circ.X(qc_box.qreg.control[0])
    circ.Ry(rotation, qc_box.qreg.control[0]).dagger()

    post_select_dict = qc_box.register_box.postselect
    post_select_dict[qc_box.qreg.control[0]] = 0
    circ_u = circuit_unitary_postselect(circ, post_select_dict)

    scipy_h = block_encoded_sparse_matrix(lcu_box).todense()

    factor = np.cos(rotation * np.pi / 2) ** 2
    scipy_u = (
        factor * scipy_h
        - (1 - factor)
        * QubitPauliOperator({QubitPauliString([Qubit(0)], [Pauli.I]): 1})
        .to_sparse_matrix(int(np.log2(scipy_h.shape[0])))
        .todense()
    )
    return circ_u, scipy_u


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_fixture"),
    ],
)
@pytest.mark.parametrize("LCUBox", [LCUMultiplexorBox])
def test_pytket_qcontrol_lcu(LCUBox: type, op: QubitPauliOperator):
    """Test the PytketQControlRegisterBix with an LCUBox."""
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    lcu_box = LCUBox(op, n_state_qubits)
    n_ancilla = 1

    clcu_box = PytketQControlRegisterBox(lcu_box, n_ancilla)

    circ_u, scipy_u = qcontrol_block_encode(clcu_box, lcu_box)
    np.testing.assert_allclose(scipy_u, circ_u, atol=1e-10)


lcu_box_test_input = LCUMultiplexorBox(
    QubitPauliOperator(
        {
            QubitPauliString([Qubit(0)], [Pauli.X]): 0.1,
            QubitPauliString([Qubit(0)], [Pauli.Y]): 0.4,
        }
    ),
    1,
)
qft_test_input = QFTBox(2)


@pytest.mark.parametrize("register_box", [lcu_box_test_input, qft_test_input])
@pytest.mark.parametrize("n_control", [1, 2, 3])
def test_pytket_qcontrol_index(register_box: LCUBox, n_control: int):
    """Test the control strong index of the .q_control().

    A test input is given for the multiplexor qcontrol method.
    and also the pytket QControlBox. the indexed controls are compared
    added to the circuit from the index input.

    Args:
    ----
        register_box (LCUBox): The LCUBox.
        n_control (int): The number of control qubits.

    """
    reg_box_unitary = register_box.get_unitary()
    for bit_index in range(2**n_control):
        bits = int_to_bits(bit_index, n_control)
        qc_reg_box = register_box.qcontrol(n_control, control_index=bit_index)

        post_pre_select: dict[Qubit, int] = dict(
            zip(qc_reg_box.qreg.control, bits, strict=True)
        )
        qc_reg_box_unitary = qc_reg_box.get_unitary(
            post_select_dict=post_pre_select, pre_select_dict=post_pre_select
        )

        np.testing.assert_allclose(reg_box_unitary, qc_reg_box_unitary, atol=1e-10)
