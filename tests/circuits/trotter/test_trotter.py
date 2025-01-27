"""Test the trotter module."""

import pytest
import numpy as np
from qtnmtts.circuits.trotter import TrotterPauliExpBox
from pytket.utils import QubitPauliOperator
from pytest_lazyfixture import lazy_fixture
from scipy.linalg import expm
from numpy.typing import NDArray
from sympy import Symbol
from qtnmtts.circuits.utils._testing import qcontrol_test


def scipy_trotterbox(
    op: QubitPauliOperator, n_state_qubits: int, time_slice: float
) -> NDArray[np.complex128]:
    """Return the scipy matrix for the TrotterPauliExpBox.

    Args:
    ----
        op (QubitPauliOperator): The hamiltonian to be approximated.
        n_state_qubits (int): The number of qubits in the state register.
        time_slice (float): The time slice of the Trotter step.

    Returns:
    -------
        np.ndarray: The scipy matrix for the TrotterPauliExpBox.

    """
    time_slice_op: QubitPauliOperator = op * time_slice  # type: ignore
    qpo_terms = [
        QubitPauliOperator({qps: coeff})
        for qps, coeff in time_slice_op._dict.items()  # type: ignore
    ]
    qpo_terms_mat = reversed(
        [qpo.to_sparse_matrix(n_state_qubits).todense() for qpo in qpo_terms]
    )
    scipy_u = np.eye(2**n_state_qubits, dtype=np.complex128)
    # Due to pytket angle convention
    qpo_terms_expmat = [expm(-1j * np.pi * 0.5 * mat) for mat in qpo_terms_mat]
    for mat in qpo_terms_expmat:
        scipy_u = scipy_u @ mat
    return scipy_u


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_hermitian_fixture"),
    ],
)
def test_trotterpauliexpbox(op: QubitPauliOperator):
    """Test the TrotterPauliExpBox."""
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    time_slice = 0.1
    trotter_box = TrotterPauliExpBox(op, n_state_qubits, time_slice)
    trotterbox_u = trotter_box.get_unitary()
    scipy_u = scipy_trotterbox(op, n_state_qubits, time_slice)
    np.testing.assert_allclose(trotterbox_u, scipy_u, atol=1e-10)

    symbol = Symbol("t")
    trotter_box = TrotterPauliExpBox(op, n_state_qubits, symbol)
    trotter_box.symbol_substitution({symbol: time_slice})
    trotterbox_u = trotter_box.get_unitary()
    scipy_u = scipy_trotterbox(op, n_state_qubits, time_slice)
    np.testing.assert_allclose(trotterbox_u, scipy_u, atol=1e-10)


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_hermitian_fixture"),
    ],
)
def test_trotter_qcontrol(op: QubitPauliOperator):
    """Test the qcontrol() method of TrotterPauliExpBox."""
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    time_slice = 0.1
    trotterbox = TrotterPauliExpBox(op, n_state_qubits, time_slice)
    qcontrol_test(trotterbox, atol=1e-10)


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_hermitian_fixture"),
    ],
)
@pytest.mark.parametrize("power", [2, 3, 4, 5])
def test_trotter_power(op: QubitPauliOperator, power: int):
    """Test the power() method of the TrotterPauliExpBox."""
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    time_slice = 0.1
    trotter_box = TrotterPauliExpBox(op, n_state_qubits, time_slice)
    trotterbox_u_scipy = trotter_box.get_unitary()
    for _ in range(power - 1):
        trotterbox_u_scipy = trotterbox_u_scipy @ trotter_box.get_unitary()

    trotter_box_circ_u = trotter_box.power(power).get_unitary()
    np.testing.assert_allclose(trotter_box_circ_u, trotterbox_u_scipy, atol=1e-10)
