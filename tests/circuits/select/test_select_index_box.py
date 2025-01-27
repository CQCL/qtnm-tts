"""SerialLCUOperator tests."""

from pytket.utils import QubitPauliOperator
from pytest_lazyfixture import lazy_fixture
from qtnmtts.circuits.lcu import SerialLCUOperator
import pytest
from qtnmtts.circuits.core import RegisterCircuit
import numpy as np
from cmath import polar
from qtnmtts.circuits.select import SelectIndexBox
from qtnmtts.circuits.index.method import IndexDefault, IndexUnaryIteration
from pytket.circuit import Qubit
from itertools import product
from numpy.typing import NDArray
from qtnmtts.operators import ising_model


@pytest.mark.parametrize(
    "hamiltonian",
    [
        lazy_fixture("op_fixture"),
    ],
)
def test_seriallcu_op_map(hamiltonian: QubitPauliOperator):
    """Test serial lcu operator op_map_list.

    This test takes the circuit for the serial lcu operators
    which have the phase include. It then generates the circuit unitary
    from these operators and multiplies it by the magnitude
    and compares it to the qubit pauli operator matrix for each term.

    Args:
    ----
        hamiltonian (QubitPauliOperator): The operator

    """
    n_state_qubits = (
        max(
            [
                p.index[0]
                for p_list in list(hamiltonian._dict.keys())
                for p in p_list.map.keys()
            ]
        )
        + 1
    )

    qpos = [
        QubitPauliOperator({qps: coeff}) for qps, coeff in hamiltonian._dict.items()
    ]

    serial = SerialLCUOperator(hamiltonian, n_state_qubits)
    mags: list[float] = []
    for coeff in hamiltonian._dict.values():
        mag, _ = polar(coeff)
        mags.append(mag)

    for ops in serial.op_map_list.values():
        for op, mag, qpo in zip(ops, mags, qpos, strict=True):
            circ = RegisterCircuit(n_state_qubits)
            circ.add_registerbox(op.box, op.targ_qreg_map)
            m_circ = circ.get_unitary() * mag
            m_qpo = qpo.to_sparse_matrix(n_state_qubits).todense()
            np.testing.assert_allclose(m_circ, m_qpo)


@pytest.mark.parametrize(
    "hamiltonian",
    [
        lazy_fixture("op_fixture"),
    ],
)
def test_select_index_box_default(hamiltonian: QubitPauliOperator):
    """Test select index box default.

    This test pre and post selects onto each index and compares it
    to the matrix of the operator term for that index

    Args:
    ----
        hamiltonian (QubitPauliOperator): The operator

    """
    n_state_qubits = (
        max(
            [
                p.index[0]
                for p_list in list(hamiltonian._dict.keys())
                for p in p_list.map.keys()
            ]
        )
        + 1
    )

    qpos = [
        QubitPauliOperator({qps: coeff}) for qps, coeff in hamiltonian._dict.items()
    ]

    mags: list[float] = []
    for coeff in hamiltonian._dict.values():
        mag, _ = polar(coeff)
        mags.append(mag)

    m_qpos = [qpo.to_sparse_matrix(n_state_qubits).todense() for qpo in qpos]

    select_box = SelectIndexBox(IndexDefault(), hamiltonian, n_state_qubits)

    bit_strings = list(product([0, 1], repeat=select_box.n_index_qubits))

    select_list: list[dict[Qubit, int]] = []
    for bit_string in bit_strings:
        post_pre_select: dict[Qubit, int] = dict(
            zip(select_box.qreg.index, bit_string, strict=True)
        )
        select_list.append(post_pre_select)

    ps_unitarys: list[NDArray[np.complex128]] = []
    for select, mag in zip(select_list, mags, strict=True):
        ps_unitary = select_box.get_unitary(
            post_select_dict=select, pre_select_dict=select
        )
        ps_unitarys.append(ps_unitary * mag)

    for m_qpo, ps_unitary in zip(m_qpos, ps_unitarys, strict=True):
        np.testing.assert_allclose(ps_unitary, m_qpo)


def test_select_index_box_unary():
    """Test select index box unary."""
    n_state_qubits = 4
    hamiltonian = ising_model(n_state_qubits, h=1.0, j=1.0)

    qpos = [
        QubitPauliOperator({qps: coeff}) for qps, coeff in hamiltonian._dict.items()
    ]

    mags: list[float] = []
    for coeff in hamiltonian._dict.values():
        mag, _ = polar(coeff)
        mags.append(mag)

    m_qpos = [qpo.to_sparse_matrix(n_state_qubits).todense() for qpo in qpos]

    select_box = SelectIndexBox(IndexUnaryIteration(), hamiltonian, n_state_qubits)

    bit_strings = list(product([0, 1], repeat=select_box.n_index_qubits))

    select_list: list[dict[Qubit, int]] = []
    work_qubits_dict = {q: False for q in select_box.qreg.work}
    for bit_string in bit_strings:
        post_pre_select: dict[Qubit, int] = dict(
            zip(select_box.qreg.index, bit_string, strict=True)
        )
        post_pre_select.update(work_qubits_dict.copy())
        select_list.append(post_pre_select)

    ps_unitarys: list[NDArray[np.complex128]] = []
    for select, mag in zip(select_list, mags, strict=False):
        ps_unitary = select_box.get_unitary(
            post_select_dict=select, pre_select_dict=select
        )
        ps_unitarys.append(ps_unitary * mag)

    for m_qpo, ps_unitary in zip(m_qpos, ps_unitarys, strict=True):
        np.testing.assert_allclose(ps_unitary, m_qpo)
