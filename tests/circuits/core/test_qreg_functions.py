"""qreg dataclass functions tests."""

from qtnmtts.circuits.core.qreg_functions import (
    extend_new_qreg_dataclass,
    make_qreg_dataclass,
)
from dataclasses import dataclass
from pytket.circuit import QubitRegister
from typing import Any
import pytest


def test_make_qreg_dataclass_single_qreg():
    """Test make_qreg_dataclass with a single qubit register."""
    qreg_dict: dict[str, QubitRegister | list[QubitRegister]] = {
        "a": QubitRegister("a", 2)
    }
    qregs = make_qreg_dataclass(qreg_dict)
    assert qregs.__annotations__ == {"a": QubitRegister}
    assert qregs.a == qreg_dict["a"]


def test_make_qreg_dataclass_multiple_qregs():
    """Test make_qreg_dataclass with multiple qubit registers."""
    qreg_dict: dict[str, QubitRegister | list[QubitRegister]] = {
        "x": QubitRegister("a", 2),
        "y": QubitRegister("b", 3),
        "z": QubitRegister("c", 4),
    }
    qregs = make_qreg_dataclass(qreg_dict)
    assert qregs.__annotations__ == {
        "x": QubitRegister,
        "y": QubitRegister,
        "z": QubitRegister,
    }
    assert qregs.x == qreg_dict["x"]
    assert qregs.y == qreg_dict["y"]
    assert qregs.z == qreg_dict["z"]


def test_make_qreg_dataclass_QubitRegister_list():
    """Test make_qreg_dataclass with a list of QubitRegister."""
    qreg_dict: dict[str, QubitRegister | list[QubitRegister]] = {
        "a": [QubitRegister("a", 2), QubitRegister("b", 3)]
    }
    qregs = make_qreg_dataclass(qreg_dict)
    assert qregs.__annotations__ == {"a": list[QubitRegister]}
    assert qregs.a == qreg_dict["a"]


@dataclass
class QubitRegisterOld:
    """Qubit Register old test."""

    qubits: QubitRegister


def test_extend_new_qreg_dataclass():
    """Test extend_new_qreg_dataclass."""
    qreg_old = QubitRegisterOld(QubitRegister("old", 3))
    extend_attrs: dict[str, QubitRegister | list[QubitRegister]] = {
        "new_qubits": QubitRegister("new", 2),
        "another_qubits": QubitRegister("another", 1),
    }
    new_qreg: Any = extend_new_qreg_dataclass(
        "QubitRegisterNew", qreg_old, extend_attrs
    )
    assert hasattr(new_qreg, "new_qubits")
    assert hasattr(new_qreg, "another_qubits")
    assert new_qreg.new_qubits == extend_attrs["new_qubits"]
    assert new_qreg.another_qubits == extend_attrs["another_qubits"]


def test_extend_new_qreg_dataclass_error():
    """Test extend_new_qreg_dataclass error."""
    qreg_old = QubitRegisterOld(QubitRegister("old", 3))
    extend_attrs: dict[str, QubitRegister | list[QubitRegister]] = {
        "qubits": QubitRegister("new", 2),
        "another_qubits": QubitRegister("another", 1),
    }
    with pytest.raises(
        ValueError, match="QubitRegister attribute qubits already exists in"
    ):
        extend_new_qreg_dataclass("QubitRegisterNew", qreg_old, extend_attrs)
