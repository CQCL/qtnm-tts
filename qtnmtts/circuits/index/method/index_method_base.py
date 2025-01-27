"""Base Class for the Index Method."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pytket.circuit import QubitRegister

from typing import Any, TYPE_CHECKING

from qtnmtts.circuits.core import RegisterCircuit

if TYPE_CHECKING:
    from qtnmtts.circuits.index import IndexOperations


@dataclass
class IndexQRegs:
    """QROMBox qubit registers.

    Attributes
    ----------
        index (QubitRegister): The address register (default - i)
        target (QubitRegister): The target register (default - t)

    """

    index: QubitRegister
    target: QubitRegister


class IndexMethodBase(ABC):
    """Base class for the index method.

    This class is an abstract class for the index method.
    It ensures that all index methods have the same interface.

    All subclasses must implement the index_circuit method.
    Which is used in composition in Box classes which use indexed operations.

    Unary iteration https://arxiv.org/abs/1805.03662,
    Dirty Qubits https://arxiv.org/abs/1812.00954 and Pytket indexed multi controlled
    operations are examples of classes which could use the index method.

    """

    @abstractmethod
    def index_circuit(
        self, indexed_ops: IndexOperations
    ) -> tuple[RegisterCircuit, Any]:
        """Abstract method for the index method."""
        pass

    @property
    @abstractmethod
    def has_work(self) -> bool:
        """Return True if the index method has a work register."""
        pass
