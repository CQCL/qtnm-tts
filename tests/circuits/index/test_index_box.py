"""Tests for the pytket index class."""

from pytket._tket.circuit import Circuit, CircBox
from pytket.circuit import Qubit, QubitRegister
from qtnmtts.circuits.core import RegisterBox, QRegMap
import numpy as np
from itertools import product
import pytest

from qtnmtts.circuits.index import IndexBox, IndexOpMap
from qtnmtts.circuits.index.method import IndexDefault


def _generate_ry_circbox(n_qubits: int):
    """Generate a random Ry circuit."""
    circ = Circuit(n_qubits)
    for n in range(n_qubits):
        circ.Ry(float(np.random.rand(1)), n)
    return CircBox(circ)


@pytest.mark.parametrize("n_index_qubits", [1, 2, 3])
@pytest.mark.parametrize("n_target_qubits", [1, 2, 3])
def test_index_pytket_box(n_index_qubits: int, n_target_qubits: int):
    """Test the IndexPytketBox."""
    # if I define the qubit register here

    index_method = IndexDefault()

    reg_box_list = [
        RegisterBox.from_CircBox(_generate_ry_circbox(n_target_qubits))
        for _ in range(2**n_index_qubits)
    ]

    target_qreg = QubitRegister("t", n_target_qubits)
    map_list = [QRegMap([reg_box.qubits], [target_qreg]) for reg_box in reg_box_list]
    op_map_list = [
        IndexOpMap(reg_box, op_map)
        for reg_box, op_map in zip(reg_box_list, map_list, strict=False)
    ]
    input_reg_dict = {target_qreg: op_map_list}

    index_box = IndexBox(index_method, input_reg_dict)

    bit_strings = list(product([0, 1], repeat=index_box.n_index_qubits))

    select_list: list[dict[Qubit, int]] = []
    for bit_string in bit_strings:
        post_pre_select: dict[Qubit, int] = dict(
            zip(index_box.qreg.index, bit_string, strict=True)
        )
        select_list.append(post_pre_select)

    for op, select in zip(reg_box_list, select_list, strict=False):
        op_unitary = op.get_unitary()
        ps_unitary = index_box.get_unitary(
            post_select_dict=select, pre_select_dict=select
        )
        np.testing.assert_allclose(op_unitary, ps_unitary, atol=1e-10)


@pytest.mark.parametrize("n_index_qubits", [1, 2])
@pytest.mark.parametrize("n_target_qubits", [2])
@pytest.mark.parametrize("n_target_registers", [2])
def test_index_pytket_box_2_target_qreg(
    n_index_qubits: int, n_target_qubits: int, n_target_registers: int
):
    """Test the IndexPytketBox."""
    from qtnmtts.circuits.index import IndexBox, IndexOpMap
    from qtnmtts.circuits.index.method import IndexDefault

    # if I define the qubit register here

    index_method = IndexDefault()

    input_reg_dict: dict[QubitRegister, list[IndexOpMap]] = {}
    regs_box_list: list[list[RegisterBox]] = []
    for n in range(n_target_registers):
        reg_box_list = [
            RegisterBox.from_CircBox(_generate_ry_circbox(n_target_qubits))
            for _ in range(2**n_index_qubits)
        ]
        regs_box_list.append(reg_box_list)
        target_qreg = QubitRegister(f"t{n}", n_target_qubits)
        map_list = [
            QRegMap([reg_box.qubits], [target_qreg]) for reg_box in reg_box_list
        ]
        op_map_list = [
            IndexOpMap(reg_box, op_map)
            for reg_box, op_map in zip(reg_box_list, map_list, strict=False)
        ]
        input_reg_dict[target_qreg] = op_map_list

    index_box = IndexBox(index_method, input_reg_dict)

    bit_strings = list(product([0, 1], repeat=index_box.n_index_qubits))

    select_list: list[dict[Qubit, int]] = []
    for bit_string in bit_strings:
        post_pre_select: dict[Qubit, int] = dict(
            zip(index_box.qreg.index, bit_string, strict=True)
        )
        select_list.append(post_pre_select)

    for op1, op2, select in zip(
        regs_box_list[0], regs_box_list[1], select_list, strict=False
    ):
        o1_unitary = op1.get_unitary()
        o2_unitary = op2.get_unitary()
        op_unitary = np.kron(o1_unitary, o2_unitary)
        ps_unitary = index_box.get_unitary(
            post_select_dict=select, pre_select_dict=select
        )
        np.testing.assert_allclose(op_unitary, ps_unitary, atol=1e-10)


def test_check_target_input():
    """Test the _check_target_input method of IndexBox."""
    index_method = IndexDefault()
    n_index_qubits = 2

    n_target_qubits = 2
    reg_box_list = [
        RegisterBox.from_CircBox(_generate_ry_circbox(n_target_qubits))
        for _ in range(2**n_index_qubits)
    ]

    target_qreg = QubitRegister("t", n_target_qubits)
    map_list = [QRegMap([reg_box.qubits], [target_qreg]) for reg_box in reg_box_list]
    op_map_list = [
        IndexOpMap(reg_box, op_map)
        for reg_box, op_map in zip(reg_box_list, map_list, strict=False)
    ]

    # Test with invalid target qubits
    input_reg_dict = {QubitRegister("d", 3): op_map_list}
    with pytest.raises(
        ValueError, match="qreg map circ qubits are not a subset of circ qubits"
    ):
        IndexBox(index_method, input_reg_dict)
