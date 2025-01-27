"""test_lcu_custom.py."""

from qtnmtts.circuits.lcu import LCUCustomBox
from qtnmtts.circuits.prepare import PrepareCircBox
from qtnmtts.circuits.select import SelectCircBox
from pytket.circuit import CircBox
from pytket._tket.circuit import Circuit
import pytest
import numpy as np


@pytest.mark.parametrize("n_prep", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("n_state", [1, 2, 3, 4, 5])
def test_lcu_custom_empty(n_prep: int, n_state: int):
    """Test LCUCustomBox is created correctly with empty circuits."""
    prep_circbox = PrepareCircBox(CircBox(Circuit(n_prep)))
    select_circbox = SelectCircBox(CircBox(Circuit(n_state + n_prep)), n_state)

    lcu_custom_box = LCUCustomBox(prep_circbox, select_circbox)

    assert isinstance(lcu_custom_box.prepare_box, PrepareCircBox)
    assert isinstance(lcu_custom_box.select_box, SelectCircBox)
    assert lcu_custom_box.n_qubits == n_prep + n_state


def test_lcu_custom_random():
    """Test random circuit is the same as the one used to create the LCUCustomBox."""
    prep_circuit = (
        Circuit(2)
        .Ry(float(np.random.rand(1)), 0)
        .Ry(float(np.random.rand(1)), 1)
        .CX(0, 1)
    )
    prep_circuit_copy = prep_circuit.copy()
    prep_circbox = CircBox(prep_circuit)

    select_circuit = (
        Circuit(4)
        .Ry(float(np.random.rand(1)), 0)
        .Ry(float(np.random.rand(1)), 1)
        .Ry(float(np.random.rand(1)), 2)
        .Ry(float(np.random.rand(1)), 3)
        .CX(0, 1)
        .CX(2, 3)
    )
    select_circuit_copy = select_circuit.copy()
    select_circbox = CircBox(select_circuit)

    circ = Circuit(4)
    circ.add_gate(prep_circbox, [0, 1])
    circ.add_gate(select_circbox, [0, 1, 2, 3])
    circ.add_gate(prep_circbox.dagger, [0, 1])

    n_state_qubits = 2
    prep_box = PrepareCircBox(CircBox(prep_circuit_copy))
    select_box = SelectCircBox(CircBox(select_circuit_copy), n_state_qubits)

    lcu_custom_box = LCUCustomBox(prep_box, select_box)

    np.testing.assert_allclose(
        lcu_custom_box.get_circuit().get_unitary(), circ.get_unitary(), atol=1e-15
    )


if __name__ == "__main__":
    test_lcu_custom_random()
