"""Index method for pytket backend."""

from __future__ import annotations
from qtnmtts.circuits.index.method import IndexMethodBase
from qtnmtts.circuits.core import RegisterCircuit, QRegMap
from pytket.circuit import QubitRegister
from typing import Any, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from qtnmtts.circuits.index import IndexOperations


@dataclass
class IndexDefaultQRegs:
    """IndexDault Method qubit registers.

    Attributes
    ----------
        index (QubitRegister): The address register (default - i)
        target (QubitRegister): The list of target registers

    """

    index: QubitRegister
    target: list[QubitRegister]


class IndexDefault(IndexMethodBase):
    """Index method for default .qcontrol() indexed controlled operators.

    This class is a subclass of IndexMethodBase.
    It is used to create indexed controlled operations for a range of binary
    integers. It used the default pytket.qcontrol() method. It is the default
    index method used in qtnmtts.

    It can index operations acros multiple registers that have the same conntrol
    index on the index register.

    Args:
    ----
        index_qreg_str (str): The index register string. Defaults to "i".
        target_qreg_str (str): The target register string. Defaults to "t".

    """

    def __init__(self):
        """Initialise the IndexPytket."""
        super().__init__()

    def index_circuit(
        self, indexed_ops: IndexOperations
    ) -> tuple[RegisterCircuit, Any]:
        """Circuit indexing method for default .qcontol().

        This method is used to create the indexed circuit using the default
        .qcontrol() method in pytket. The indexed operations are stored in the
        IndexOperations object. They are then looped over and the multi control
        strings are applied.

        This returned output will going into the RegisterBox Base class. Where this
        class is to be passed into the IndexBox class and is to be used in composition.

        Args:
        ----
            indexed_ops (IndexOperations): The indexed operations.

        """
        circ = RegisterCircuit(self.__class__.__name__)

        index_qreg = circ.add_q_register(indexed_ops.index_qreg)
        for t_qreg in indexed_ops.target_q_registers:
            circ.add_q_register(t_qreg)

        self._qregs = IndexDefaultQRegs(index_qreg, indexed_ops.target_q_registers)

        for i, operation in enumerate(indexed_ops.index):
            for reg_operation in operation.op_map_reg:
                qc_box = reg_operation.box.qcontrol(
                    indexed_ops.n_index_qubits, control_index=i
                )
                qreg_map = QRegMap(
                    [qc_box.qreg.control, reg_operation.targ_qreg_map.box_qubits],
                    [index_qreg, reg_operation.targ_qreg_map.circ_qubits],
                )
                circ.add_registerbox(qc_box, qreg_map)

        return circ, self._qregs

    @property
    def has_work(self) -> bool:
        """Return if the index method has work qubits."""
        return False
