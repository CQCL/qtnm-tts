"""Module for SelectIndex operators."""

from pytket.circuit import Op, QubitRegister, Qubit, OpType
from pytket.pauli import Pauli, QubitPauliString
from pytket._tket.circuit import Circuit, Unitary1qBox
from qtnmtts.circuits.index import IndexOpMap
from pytket.utils.operators import QubitPauliOperator
from qtnmtts.circuits.core import RegisterBox, QRegMap
import cmath


class SerialLCUOperator:
    """Generates the op_map_list for LCU for a QubitPauliOperator.

    Each term of Paulis in the QubitPauliOperator is converted to a list of phased
    pytket Ops. Each term in the QubitPauliOperator is then converted to
    a RegisterBox. The op_map_list is a dictionary with the state register
    as the key and a list of IndexOpMap for each tome as the value.
    Each IndexOpMap contains the RegisterBox and the QRegMap for the term.

    The term is stored as a list of pytket Ops and a magnitude.
    The phase of the coeff of the term is absorbed into the pauli Op making it
    a general SU(2) Op. This is because in LCU the Prepare state can only be
    positive magnitudes, so the phase is absorbed into the LCU for each term

    Args:
    ----
        hamiltonian (QubitPauliOperator): The operator to be applied.
        n_state_qubits (int): The number of qubits in the state register.

    """

    def __init__(self, hamiltonian: QubitPauliOperator, n_state_qubits: int):
        """Initialise the SerialLCUOperator Class."""
        terms_ops: list[list[Op]] = [
            self.pauli_ops(term, coeff)  # type: ignore
            for term, coeff in hamiltonian._dict.items()
        ]
        terms_qubits: list[list[Qubit]] = [
            self.term_qubits(term) for term in hamiltonian._dict.keys()
        ]

        self._is_hermitian = all(
            self._is_term_hermitian(coeff)  # type: ignore
            for coeff in hamiltonian._dict.values()
        )
        self._op_map_list = self._op_map_list(terms_ops, terms_qubits, n_state_qubits)  # type: ignore

    @property
    def op_map_list(self) -> dict[QubitRegister, list[IndexOpMap]]:
        """Return the op_map_list."""
        return self._op_map_list  # type: ignore

    @property
    def is_hermitian(self) -> bool:
        """Return True if the operator is hermitian."""
        return self._is_hermitian

    def _term_to_registerbox(self, term_ops: list[Op]) -> RegisterBox:
        """Convert a term to a RegisterBox."""
        circ = Circuit(len(term_ops))
        for q, op in enumerate(term_ops):
            circ.add_gate(op, [q])
        return RegisterBox.from_Circuit(circ)

    def _op_map_list(
        self,
        terms_ops: list[list[Op]],
        terms_qubits: list[list[Qubit]],
        n_state_qubits: int,
    ) -> dict[QubitRegister, list[IndexOpMap]]:
        """Convert phased pauli op to RegisterBox.

        Args:
        ----
            terms_ops (list[list[Op]]): The list of ops for each term.
            terms_qubits (list[list[Qubit]]): The list of qubits in each term.
            n_state_qubits (int): The number of qubits in the state register.

        Returns:
        -------
            dict[QubitRegister, list[IndexOpMap]]: The op_map_list.

        """
        op_map_list: list[IndexOpMap] = []
        for term_ops, term_qubits in zip(terms_ops, terms_qubits, strict=True):
            reg_box = self._term_to_registerbox(term_ops)
            op_map_list.append(
                IndexOpMap(reg_box, QRegMap([reg_box.qubits], [term_qubits]))
            )
        return {QubitRegister("q", n_state_qubits): op_map_list}

    def pauli_ops(self, term: QubitPauliString, coeff: complex) -> list[Op]:
        """Convert term Paulis to a list of phased pytket Ops.

        The phase of the coeff of the term is absorbed into
        the first pauli Op making as a general Unitary1qBox.
        If the term is empty then the identity is returned.

        Args:
        ----
            term (QubitPauliString): The term to be applied.
            coeff (complex): The coefficient of the term.

        Returns:
        -------
            list[Op]: The list of phased pytket Ops.

        """
        ops = {
            Pauli.I: Op.create(OpType.noop),
            Pauli.X: Op.create(OpType.X),
            Pauli.Y: Op.create(OpType.Y),
            Pauli.Z: Op.create(OpType.Z),
        }

        paulis = list(term.map.values())

        if paulis == []:
            paulis = [Pauli.I]

        pauli_ops = [ops[pauli] for pauli in paulis]

        _, phase = cmath.polar(coeff)
        exp = cmath.exp(phase * 1j)

        pauli_ops[0] = Unitary1qBox(pauli_ops[0].get_unitary() * exp)

        return pauli_ops

    def term_qubits(self, term: QubitPauliString) -> list[Qubit]:
        """Return the qubits in the term."""
        qubits = list(term.map.keys())
        if qubits == []:
            qubits = [Qubit(0)]
        return qubits

    def _is_term_hermitian(self, coeff: complex) -> bool:
        """Return True if the term is hermitian."""
        _, phase = cmath.polar(coeff)
        exp = cmath.exp(phase * 1j)

        if cmath.isclose(exp, 1) or cmath.isclose(exp, -1):
            is_hermitian = True
        else:
            is_hermitian = False
        return is_hermitian
