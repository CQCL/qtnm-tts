"""Tests for the ReflectionBox implementation."""

import pytest
import numpy as np
from qtnmtts.circuits.reflection import ReflectionBox
from qtnmtts.circuits.core import PytketQControlRegisterBox
from qtnmtts.circuits.reflection.reflection_registerbox import QControlReflectionBox


@pytest.mark.parametrize("n_qubits", np.arange(1, 7))
def test_reflection_box(n_qubits: int):
    """Test the ReflectionBox.

    Test the unitaries of the ReflectionBox compared to the expected
    unitary as well as the signs.

    Args:
    ----
        n_qubits (int): number of qubits

    """
    reflection_box = ReflectionBox(n_qubits)
    reflection_box_neg = ReflectionBox(n_qubits, positive=False)

    reflection_unitary = -1 * np.diag(np.ones(2**n_qubits))
    reflection_unitary[0, 0] = 1

    reflection_unitary_neg = -1 * reflection_unitary
    np.testing.assert_allclose(
        reflection_box.get_unitary(), reflection_unitary, atol=1e-10
    )
    np.testing.assert_allclose(
        reflection_box_neg.get_unitary(), reflection_unitary_neg, atol=1e-10
    )


@pytest.mark.parametrize("n_qubits", np.arange(1, 7))
def test_reflection_box_qcontrol(n_qubits: int):
    """Test the ReflectionBox.qcontrol(1) method.

    Compare for both signs that the unitary is equivalent as the one using
    PytketQControlRegisterBox.

    Args:
    ----
        n_qubits (int): number of qubits

    """
    reflection_box = ReflectionBox(n_qubits)
    reflection_box_neg = ReflectionBox(n_qubits, positive=False)

    reflection_box_qcontrol = reflection_box.qcontrol(1)
    reflection_box_neg_qcontrol = reflection_box_neg.qcontrol(1)

    assert isinstance(reflection_box_qcontrol, QControlReflectionBox)
    assert isinstance(reflection_box_neg_qcontrol, QControlReflectionBox)

    reflection_unitary_qcontrol = PytketQControlRegisterBox(reflection_box, 1)
    reflection_unitary_qcontrol_neg = PytketQControlRegisterBox(reflection_box_neg, 1)

    np.testing.assert_allclose(
        reflection_box_qcontrol.get_unitary(),
        reflection_unitary_qcontrol.get_unitary(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        reflection_box_neg_qcontrol.get_unitary(),
        reflection_unitary_qcontrol_neg.get_unitary(),
        atol=1e-10,
    )
