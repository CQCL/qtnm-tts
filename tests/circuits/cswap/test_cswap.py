"""Tests for the amplitude amplification circuit implementation."""

import pytest
import numpy as np
from numpy.typing import NDArray


from qtnmtts.circuits.core import RegisterBox

from pytket.circuit import StatePreparationBox
from qtnmtts.circuits.cswap import CSWAPRegisterBox

from pytket.circuit import QubitRegister, Qubit

from qtnmtts.circuits.utils.linalg import kron_list

# from pytket.utils.operators import QubitPauliOperator


def swap_states(state0: NDArray[np.complex128], statef: NDArray[np.complex128]):
    """Swap two states."""
    return np.kron(statef, state0)


def assert_cswap_box(n_state_qubits: int, n_registers: int):
    """Test that two states are swapped or not.

    Args:
    ----
        n_state_qubits (int): no. of state qubits.
        n_registers (int): no. of registers.

    """
    np.random.seed(3)

    def get_vec():
        np.random.seed(3)
        state0 = np.random.rand(2**n_state_qubits) + (
            1j * np.random.rand(2**n_state_qubits)
        )
        return state0 / np.sqrt(state0.conj().T @ state0)

    states_0 = [get_vec() for _ in range(n_registers)]
    states_1 = [get_vec() for _ in range(n_registers)]

    state_boxes_0 = [StatePreparationBox(v) for v in states_0]
    state_boxes_1 = [StatePreparationBox(v) for v in states_1]

    control_qreg = Qubit("c", 0)
    a_qregs = [QubitRegister(f"a{i}", n_state_qubits) for i in range(n_registers)]
    b_qregs = [QubitRegister(f"b{i}", n_state_qubits) for i in range(n_registers)]

    cswap_box = CSWAPRegisterBox(control_qreg, a_qregs, b_qregs)
    cswap_circ1 = cswap_box.initialise_circuit()

    cswap_circ1.X(control_qreg)
    for i, (state_a, state_b) in enumerate(
        zip(state_boxes_0, state_boxes_1, strict=False)
    ):
        cswap_circ1.add_gate(state_a, cswap_box.qreg.a[i].to_list())
        cswap_circ1.add_gate(state_b, cswap_box.qreg.b[i].to_list())

    cswap_circ1.add_registerbox(cswap_box)  # NO map needed as same qregs

    cswap_box1 = RegisterBox.from_Circuit(cswap_circ1)
    cswap1_box_state = cswap_box1.get_statevector({control_qreg: 1})

    state_0_kron = kron_list(states_0)
    state_1_kron = kron_list(states_1)

    tensor_prod = swap_states(state_0_kron, state_1_kron)
    np.testing.assert_allclose(cswap1_box_state, tensor_prod, atol=1e-10)


@pytest.mark.parametrize("n_registers", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("n_state_qubits", [1])
def test_cswap_box_1q(n_state_qubits: int, n_registers: int):
    """Test that two states are swapped or not."""
    assert_cswap_box(n_state_qubits, n_registers)


@pytest.mark.parametrize("n_registers", [1, 2])
@pytest.mark.parametrize("n_state_qubits", [2])
def test_cswap_box_2q(n_state_qubits: int, n_registers: int):
    """Test that two states are swapped or not."""
    assert_cswap_box(n_state_qubits, n_registers)


@pytest.mark.parametrize("n_registers", [1])
@pytest.mark.parametrize("n_state_qubits", [3, 4, 5])
def test_cswap_box_3q(n_state_qubits: int, n_registers: int):
    """Test that two states are swapped or not."""
    assert_cswap_box(n_state_qubits, n_registers)
