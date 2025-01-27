"""IndexBox class for indexed operations."""

from qtnmtts.circuits.core import RegisterBox, QRegMap, RegisterCircuit
from pytket.circuit import QubitRegister
from qtnmtts.circuits.index.method import IndexMethodBase
from math import ceil, log2
from dataclasses import dataclass


"""IndexBox Baseclass for Indexed operations."""


@dataclass
class IndexOpMap:
    """IndexOMap data class.

    This class is used to store the indexed operation and the QRegMap that maps the box
    qubit to the target qubits in the indexed operations. A pytket CircBox must be
    converted to a RegisterBox because the RegisterBox().qcontrol() is used
    in the indexed circuit.

    Attributes
    ----------
    box (RegisterBox): The box to be controlled indexed.
    target_list (Qubit): The target qubits which the operation acts on.

    """

    box: RegisterBox
    targ_qreg_map: QRegMap

    def __post_init__(self):
        """Post init method for the IndexOp."""
        if self.box.n_qubits != len(self.targ_qreg_map.box_qubits):
            raise ValueError(
                "The number of qubits in the box does not match the \
                             number of qubits in the qreg map"
            )

        if self.box.qubits != self.targ_qreg_map.box_qubits:
            raise ValueError("The box qubits must match the qreg map box qubits")


@dataclass
class IndexOpMapRegs:
    """IndexOpMapRegs data class.

    This class stores a list of IndexOpMap objects that have the same index,
    but act on different qubit registers.

    Attributes
    ----------
        op_map_reg (list[IndexOpMap]): The list of IndexOpMap objects for
            a single qubit register

    """

    op_map_reg: list[IndexOpMap]


class IndexOperations:
    """IndexOperations for multiple registers class.

    This class stores the indexed operations for multiple qubit registers
    where the indexed operations across multiple registers act on the
         same control string.

    Attributes
    ----------
        register_ops (dict[QubitRegister, list[IndexOpMap]]): The dictionary of
          QubitRegister and IndexOpMap objects.
        index_qreg_str (str): The index register string. Defaults to "i"

    """

    def __init__(
        self,
        register_ops: dict[QubitRegister, list[IndexOpMap]],
        index_qreg_str: str = "i",
    ):
        """Initialize the IndexOperations."""
        self._index_qreg_str = index_qreg_str

        n_index = self._check_input(register_ops)
        self._register_ops = register_ops
        self._q_registers = list(register_ops.keys())

        self._index: list[IndexOpMapRegs] = []
        for i in range(n_index):
            list_data: list[IndexOpMap] = []
            for op_map in register_ops.values():
                list_data.append(op_map[i])
            self._index.append(IndexOpMapRegs(list_data))

    def _check_input(self, register_ops: dict[QubitRegister, list[IndexOpMap]]):
        """Check the register data.

        Check the input data across all registers to ensure that all the target qreg
        operations have the same length.

        Args:
        ----
            register_ops (dict[QubitRegister, list[IndexOpMap]]): The dictionary of
            QubitRegister and IndexOpMap objects.

        Returns:
        -------
            int: The length of the operations.

        """
        len_data = len(next(iter(register_ops.values())))
        if not all(len(sublist) == len_data for sublist in register_ops.values()):
            raise ValueError("All target qreg operations must have the same length")
        return len_data

    @property
    def index(self) -> list[IndexOpMapRegs]:
        """Return the index."""
        return self._index

    @property
    def n_index_qubits(self) -> int:
        """Return the number of index qubits."""
        return int(ceil(log2(len(self.index))))

    @property
    def index_qreg(self):
        """Return the index qubit register."""
        return QubitRegister(self._index_qreg_str, self.n_index_qubits)

    @property
    def n_index(self) -> int:
        """Return the data length."""
        return len(self.index)

    @property
    def target_q_registers(self) -> list[QubitRegister]:
        """Return the registers."""
        return self._q_registers

    @property
    def register_ops(self) -> dict[QubitRegister, list[IndexOpMap]]:
        """Return the data."""
        return self._register_ops

    @property
    def index_bools(self) -> list[list[bool]]:
        """Return the index boolean list."""
        from qtnmtts.circuits.utils import int_to_bits

        return [int_to_bits(i, self.n_index_qubits) for i in range(self.n_index)]


class IndexBox(RegisterBox):
    """IndexBox base class.

    This class inherits from RegisterBox. It is the class for indexed operations
    where an index register has a set of multi-controlled operations acting on
    multiple target registers.

    The type of indexed operation is defined by index_method.

    Args:
    ----
        index_method (IndexMethodBase): The index method.
        op_map_regs (dict[QubitRegister, list[IndexOpMap]]): The dictionary of
            QubitRegister and IndexOpMap objects.
        index_qreg_str (str): The index register string. Defaults to "i".

    """

    def __init__(
        self,
        index_method: IndexMethodBase,
        op_map_regs: dict[QubitRegister, list[IndexOpMap]],
        index_qreg_str: str = "i",
    ):
        """Initialize the IndexBox."""
        self._indexed_ops = IndexOperations(op_map_regs, index_qreg_str)
        self._op_map_regs = op_map_regs

        self._index_method = index_method
        circ, qregs = self._index_method.index_circuit(self._indexed_ops)

        super().__init__(qregs, circ)

    @property
    def n_index_qubits(self) -> int:
        """Return the number of index qubits."""
        return self._indexed_ops.n_index_qubits

    @property
    def indexed_ops(self) -> IndexOperations:
        """Return the indexed operations."""
        return self._indexed_ops

    @property
    def op_map_regs(self) -> dict[QubitRegister, list[IndexOpMap]]:
        """Return the indexed op map for each register."""
        return self._op_map_regs

    @property
    def index_method(self) -> IndexMethodBase:
        """Return the index method."""
        return self._index_method

    def _check_target_input(self, op_map_regs: dict[QubitRegister, list[IndexOpMap]]):
        """Check the target input."""
        for qreg, op_map_list in op_map_regs.items():
            for op_map in op_map_list:
                for q in op_map.targ_qreg_map.circ_qubits:
                    if q not in qreg:
                        raise ValueError(
                            "The target qubits must be \
                                           in the target register"
                        )

    def initialise_circuit(self, no_index: bool = False) -> RegisterCircuit:
        """Initialise the circuit.

        Initialise the circuit for the IndexBox.

        Args:
        ----
            no_index (bool): Whether to include the index qubits in the circuit.

        Returns:
        -------
            RegisterCircuit: The initialised circuit.

        """
        qreg_circ = RegisterCircuit()
        if not no_index:
            for qreg in self.q_registers:
                qreg_circ.add_q_register(qreg)
        else:
            for qreg in self.q_registers:
                if qreg != self.qreg.index:
                    qreg_circ.add_q_register(qreg)
        return qreg_circ
