"""Tests for the amplitude amplification circuit implementation."""

import pytest
import numpy as np
from numpy.typing import NDArray

from qtnmtts.measurement.utils import circuit_unitary_postselect
from qtnmtts.circuits.lcu import LCUMultiplexorBox
from qtnmtts.circuits.core import QRegMap
from qtnmtts.circuits.amplitude_amplification._amplification_registerbox import (
    AmplificationBox,
)
from pytket.utils.operators import QubitPauliOperator

from pytest_lazyfixture import lazy_fixture


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_fixture"),
    ],
)
def success_probability(op: QubitPauliOperator, state0: NDArray[np.complex128]):
    "The theoretical success probability and amplified one." ""
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )
    op_mat = op.to_sparse_matrix(n_state_qubits).todense()
    statef = op_mat @ state0

    alpha = sum(np.abs([np.complex128(val) for val in op._dict.values()]))

    lcu_success_probability = np.abs((statef @ statef.conj().T) / np.abs(alpha) ** 2)

    statef_amp = (
        (3 * op_mat) - ((4 / alpha**2) * op_mat @ op_mat.conj().transpose() @ op_mat)
    ) @ state0
    success_probability_amp = np.abs((statef_amp @ statef_amp.conj().T) / alpha**2)

    return lcu_success_probability.item((0, 0)), success_probability_amp.item((0, 0))


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_fixture"),
    ],
)
@pytest.mark.parametrize("AmplificationBox", [AmplificationBox])
def test_amp_op(AmplificationBox: type, op: QubitPauliOperator):
    """Test that operator obtained from postselecting the LCU.

    Circuit Must be the same as the block encoded operator itself.

    Args:
    ----
        AmplificationBox: The LCUBox to test.
        op (QubitPauliOperator): The operator to test.

    """
    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )

    state0 = np.random.rand(2**n_state_qubits) + (
        1j * np.random.rand(2**n_state_qubits)
    )
    state0 = state0 / np.sqrt(state0.conj().T @ state0)

    lcu_success_prob, amp_lcu_success_prob = success_probability(op, state0)

    lcu_box = LCUMultiplexorBox(op, n_state_qubits)
    circ_h = circuit_unitary_postselect(lcu_box.get_circuit(), lcu_box.postselect)

    statef = circ_h @ state0
    p = np.abs(statef.conj().T @ statef)
    np.testing.assert_allclose(lcu_success_prob, p, atol=1e-10)

    lcu_amp_circ = lcu_box.initialise_circuit()
    qreg_map = QRegMap(lcu_box.q_registers, lcu_amp_circ.q_registers)
    lcu_amp_circ.add_registerbox(lcu_box, qreg_map)
    amp_box = AmplificationBox(lcu_box, 1)
    qreg_map2 = QRegMap(amp_box.q_registers, lcu_amp_circ.q_registers)
    lcu_amp_circ.add_registerbox(amp_box, qreg_map2)
    amp_circ_h = circuit_unitary_postselect(lcu_amp_circ, lcu_box.postselect)
    statef_amp = amp_circ_h @ state0
    p_amp = np.abs(statef_amp.conj().T @ statef_amp)
    np.testing.assert_allclose(amp_lcu_success_prob, p_amp, atol=1e-10)
