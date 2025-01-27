"""Tests for the pytket index class."""

from pytket._tket.circuit import Circuit, CircBox
from pytket.circuit import Qubit, QubitRegister
from qtnmtts.circuits.core import RegisterBox, QRegMap
import numpy as np
from itertools import product
import pytest

from qtnmtts.circuits.index import IndexBox, IndexOpMap
from qtnmtts.circuits.index.method import IndexUnaryIteration, IndexMethodBase


def _generate_ry_circbox(n_qubits: int):
    """Generate a random Ry circuit."""
    circ = Circuit(n_qubits)
    for n in range(n_qubits):
        circ.Ry(float(np.random.rand(1)), n)
    return CircBox(circ)


def _index_box_assert(
    index_method: IndexMethodBase,
    n_target_qubits: int,
    reg_box_list_list: list[list[RegisterBox]],
):
    input_reg_dict: dict[QubitRegister, list[IndexOpMap]] = {}

    for i, reg_box_list in enumerate(reg_box_list_list):
        target_qreg = QubitRegister(f"t{i}", n_target_qubits)
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
    work_qubits_dict = {q: False for q in index_box.qreg.work}
    for bit_string in bit_strings:
        post_pre_select: dict[Qubit, int] = dict(
            zip(index_box.qreg.index, bit_string, strict=True)
        )
        post_pre_select.update(work_qubits_dict.copy())
        select_list.append(post_pre_select)

    if len(reg_box_list_list) == 1:
        for op, select in zip(reg_box_list_list[0], select_list, strict=False):
            op_unitary = op.get_unitary()
            ps_unitary = index_box.get_unitary(
                post_select_dict=select, pre_select_dict=select
            )
            np.testing.assert_allclose(op_unitary, ps_unitary, atol=1e-10)

    if len(reg_box_list_list) == 2:
        for op1, op2, select in zip(
            reg_box_list_list[0], reg_box_list_list[1], select_list, strict=False
        ):
            o1_unitary = op1.get_unitary()
            o2_unitary = op2.get_unitary()
            op_unitary = np.kron(o1_unitary, o2_unitary)
            ps_unitary = index_box.get_unitary(
                post_select_dict=select, pre_select_dict=select
            )
            np.testing.assert_allclose(op_unitary, ps_unitary, atol=1e-10)


@pytest.mark.parametrize("n_index_qubits", [2, 3])
@pytest.mark.parametrize("n_target_qubits", [1, 2, 3])
def test_index_unary_iteration_box(n_index_qubits: int, n_target_qubits: int):
    """Test Index Unary iteration for various qubit numbers."""
    index_method = IndexUnaryIteration()

    reg_box_list = [
        RegisterBox.from_CircBox(_generate_ry_circbox(n_target_qubits))
        for _ in range(2**n_index_qubits)
    ]

    _index_box_assert(index_method, n_target_qubits, [reg_box_list])


@pytest.mark.parametrize("n_elements", [5, 6, 7])
def test_index_unary_iteration_box_not_full(n_elements: int):
    """Test Index Unary for a non full set of elements."""
    n_target_qubits = 1

    index_method = IndexUnaryIteration()

    reg_box_list = [
        RegisterBox.from_CircBox(_generate_ry_circbox(n_target_qubits))
        for _ in range(n_elements)
    ]

    _index_box_assert(index_method, n_target_qubits, [reg_box_list])


def test_unary_iteration_2_target():
    """Test Index Unary iteration for 2 registers registers."""
    n_index_qubits = 2
    n_target_qubits = 2

    index_method = IndexUnaryIteration()

    def generate_reg_box_list(n_target_qubits: int, n_index_qubits: int):
        return [
            RegisterBox.from_CircBox(_generate_ry_circbox(n_target_qubits))
            for _ in range(2**n_index_qubits)
        ]

    reg_box_list_list = [
        generate_reg_box_list(n_target_qubits, n_index_qubits) for _ in range(2)
    ]

    _index_box_assert(index_method, n_target_qubits, reg_box_list_list)


@pytest.mark.parametrize("n_index_qubits", [2, 3, 4])
def test_unary_iteration_custom_toffoli(n_index_qubits: int):
    """Test Index Unary iteration custom toffili."""
    circ = Circuit(3)
    circ.H(2)
    circ.CX(1, 2)
    circ.Tdg(2)
    circ.CX(0, 2)
    circ.T(2)
    circ.CX(1, 2)
    circ.Tdg(2)
    circ.CX(0, 2)
    circ.T(1)
    circ.T(2)
    circ.CX(0, 1)
    circ.H(2)
    circ.T(0)
    circ.Tdg(1)
    circ.CX(0, 1)

    circ = CircBox(circ)

    n_target_qubits = 1

    index_method = IndexUnaryIteration(toffoli=circ)

    reg_box_list = [
        RegisterBox.from_CircBox(_generate_ry_circbox(n_target_qubits))
        for _ in range(2**n_index_qubits)
    ]

    _index_box_assert(index_method, n_target_qubits, [reg_box_list])
