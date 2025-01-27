"""Tests for the QFT module."""

import numpy as np
from qtnmtts.circuits.qft.standard_qft import QFTBox
import pytest
from qtnmtts.circuits.utils._testing import qft_unitary


@pytest.mark.parametrize("n_qubits", [3, 4, 5, 6])
def test_qft_circuit(n_qubits: int) -> None:
    """Test that the QFT circuit is correct for n_qubits."""
    qft_box = QFTBox(n_qubits)
    assert qft_box.get_circuit().name == "QFT"
    circ_unitary = qft_box.get_unitary()
    test_matrix = qft_unitary(n_qubits)
    np.testing.assert_allclose(circ_unitary, test_matrix, atol=1e-10)


@pytest.mark.parametrize("n_qubits", [3, 4, 5, 6])
def test_inverse_qft(n_qubits: int) -> None:
    """Test that the inverse QFT circuit is correct for 4 qubits."""
    inv_unitary = QFTBox(n_qubits).dagger.get_unitary()
    test_matrix_inv = qft_unitary(n_qubits).conj().T
    np.testing.assert_allclose(inv_unitary, test_matrix_inv, atol=1e-10)


@pytest.mark.parametrize("do_swaps", [True, False])
def test_qft_swaps(do_swaps: bool) -> None:
    """Test whether the QFTBox do swaps or not."""
    n_qubits = 3
    qft_box = QFTBox(n_qubits, do_swaps=do_swaps)
    assert qft_box.has_swaps == do_swaps
