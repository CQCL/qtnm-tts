"""Test OracleCircuit class."""

from qtnmtts.circuits.core import RegisterCircuit, QRegMap
from pytket.circuit import Qubit
from pytket._tket.circuit import Circuit
from pytket.circuit import QubitRegister
from qtnmtts.circuits.lcu import LCUMultiplexorBox
from qtnmtts.operators import ising_model
import pytest


def _get_qreg_pairs(
    n_fregisters: int, n_gregisters: int, f_size: int, g_size: int
) -> tuple[list[QubitRegister], list[QubitRegister]]:
    circ_f = Circuit()
    circ_g = Circuit()
    f_qregs = [circ_f.add_q_register(f"f_qreg{i}", f_size) for i in range(n_fregisters)]
    g_qregs = [circ_g.add_q_register(f"g_qreg{i}", g_size) for i in range(n_gregisters)]
    return f_qregs, g_qregs


@pytest.mark.parametrize("n_registers", [1, 2, 3])
@pytest.mark.parametrize("f_size", [1, 2])
@pytest.mark.parametrize("g_size", [3, 4])
def test_qreg_map_size_error(n_registers: int, f_size: int, g_size: int):
    """Test QRegMap class for not same size exception."""
    f_qregs, g_qregs = _get_qreg_pairs(n_registers, n_registers, f_size, g_size)

    f_qubits_list: list[list[Qubit]] = []
    g_qubits_list: list[list[Qubit]] = []
    for f, g in zip(f_qregs, g_qregs, strict=True):
        fq_list = f.to_list()
        gq_list = g.to_list()
        f_qubits_list.append(fq_list)
        g_qubits_list.append(gq_list)

    with pytest.raises(ValueError, match="not the same size"):
        QRegMap(f_qregs, g_qregs)

    with pytest.raises(ValueError, match="not the same size"):
        QRegMap(f_qubits_list, g_qubits_list)


def test_qreg_map_duplicate_error():
    """Test QRegMap class for not same size exception."""
    f_qregs, g_qregs = _get_qreg_pairs(2, 2, 2, 2)

    with pytest.raises(ValueError, match="appears more than once"):
        QRegMap([f_qregs[0], f_qregs[0]], g_qregs)


@pytest.mark.parametrize(
    "string0",
    ["state", "psi"],
)
@pytest.mark.parametrize("string1", ["a", "b"])
def test_add_registerbox(string0: str, string1: str):
    """Test add_registerbox method."""
    init_circ = RegisterCircuit()
    x = init_circ.add_q_register(string0, 3)
    y = init_circ.add_q_register(string1, 3)

    n_state_qubits = 3
    h = 1
    j = 1
    ising_model_3q = ising_model(3, h, j)

    lcu_box = LCUMultiplexorBox(ising_model_3q, n_state_qubits)

    # Test from list QubitRegister
    circ = init_circ.copy()
    map = QRegMap(lcu_box.q_registers, circ.q_registers)
    circ.add_registerbox(lcu_box, map)

    assert circ.qubits == [*y, *x]

    # Test from dict - QubitRegister
    circ = init_circ.copy()
    map = QRegMap.from_dict(
        dict(zip(lcu_box.q_registers, circ.q_registers, strict=True))
    )
    circ.add_registerbox(lcu_box, map)

    assert circ.qubits == [*y, *x]

    # Test from list[Qubit]
    circ = init_circ.copy()

    box_q_list = [qreg.to_list() for qreg in lcu_box.q_registers]
    circ_q_list = [qreg.to_list() for qreg in circ.q_registers]

    circ = init_circ.copy()
    map = QRegMap(box_q_list, circ_q_list)
    circ.add_registerbox(lcu_box, map)
    assert circ.qubits == [*y, *x]


def test_add_registerbox_noqregmap():
    """Test add_registerbox method without QReg."""
    n_state_qubits = 3
    h = 1
    j = 1
    ising_model_3q = ising_model(3, h, j)

    lcu_box = LCUMultiplexorBox(ising_model_3q, n_state_qubits)

    # adds to a subset of the circuit qubits with the same name
    circ = lcu_box.initialise_circuit()
    circ.add_q_register("ancilla", 3)

    assert isinstance(circ.add_registerbox(lcu_box), RegisterCircuit)

    # test fails if qubits are not a subset of the circuit qubits
    init_circ = RegisterCircuit()
    init_circ.add_q_register("prepare", 3)
    init_circ.add_q_register("psi", 3)

    with pytest.raises(
        ValueError,
        match="register_box qubits are not a subset of circuit qubits",
    ):
        init_circ.add_registerbox(lcu_box)


@pytest.mark.parametrize(
    "string0",
    ["state", "psi"],
)
@pytest.mark.parametrize("string1", ["a", "b"])
def test_add_registerbox_subset_error(string0: str, string1: str):
    """Test add_registerbox method errors."""
    init_circ = RegisterCircuit()
    init_circ.add_q_register(string0, 3)
    init_circ.add_q_register(string1, 3)

    error_circ = RegisterCircuit()
    m = error_circ.add_q_register("m", 3)
    n = error_circ.add_q_register("n", 3)

    n_state_qubits = 3
    h = 1
    j = 1
    ising_model_3q = ising_model(3, h, j)

    lcu_box = LCUMultiplexorBox(ising_model_3q, n_state_qubits)

    circ = init_circ.copy()
    map = QRegMap([m, n], circ.q_registers)
    with pytest.raises(ValueError, match="qreg map box qubits are not a subset"):
        circ.add_registerbox(lcu_box, map)

    circ = init_circ.copy()
    map = QRegMap(lcu_box.q_registers, [m, n])
    with pytest.raises(ValueError, match="qreg map circ qubits are not a subset"):
        circ.add_registerbox(lcu_box, map)


def test_qreg_map_from_QRegMap_list():
    """Test QRegMap from_QRegMap_list classmethod."""
    qreg_map_list = [
        QRegMap([QubitRegister("box_qreg1", 3)], [QubitRegister("circ_qreg1", 3)]),
        QRegMap([QubitRegister("box_qreg2", 2)], [QubitRegister("circ_qreg2", 2)]),
    ]
    qreg_map = QRegMap.from_QRegMap_list(qreg_map_list)
    assert qreg_map.box_qubits == [
        *qreg_map_list[0].box_qubits,
        *qreg_map_list[1].box_qubits,
    ]
    assert qreg_map.circ_qubits == [
        *qreg_map_list[0].circ_qubits,
        *qreg_map_list[1].circ_qubits,
    ]
