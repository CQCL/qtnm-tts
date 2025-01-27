"""initialise the TrotterBox class."""

from dataclasses import dataclass
from qtnmtts.circuits.core import RegisterBox, RegisterCircuit
from pytket.circuit import PauliExpBox, QubitRegister
from pytket.pauli import QubitPauliString
from pytket.utils import QubitPauliOperator
from sympy import Symbol


@dataclass
class TrotterQReg:
    """TrotterBox qubit registers.

    Attributes
    ----------
        state (QubitRegister): The state register (default - q)

    """

    state: QubitRegister


class PauliTerm:
    """PauliTerm class.

    A class to store a Pauli string and its coefficient.
    Its qubit indices are stored in a list.

    Args:
    ----
        pauli_string (QubitPauliString): A Pauli string.
        coeff (float): The coefficient of the Pauli string.

    Attributes:
    ----------
        qubits (list[Qubit]): A list of the qubits in the Pauli string.
        q_inds (list[int]): A list of the qubit indices in the Pauli string.
        paulis (list[Pauli]): A list of the Pauli operators in the Pauli string.
        coeff (float): The coefficient of the Pauli string.

    """

    def __init__(self, pauli_string: QubitPauliString, coeff: float | Symbol):
        """Initialise the PauliTerm class."""
        self.qubits = list(pauli_string.map.keys())
        self.q_inds = [q.index[0] for q in self.qubits]
        self.paulis = list(pauli_string.map.values())
        self.coeff = coeff


class TrotterPauliExpBox(RegisterBox):
    r"""TrotterPauliExpBox class.

    A class to implement the first order Trotter approximation of the time evolution
    operator. For a single Trotter step, the time evolution operator is approximated
    by the product of the exponentials of the individual terms in the Hamiltonian.
    The TrotterPauliExpBox class implements this approximation by applying the
    PauliExpBox gate for each term in the Hamiltonian.

    $$ e^{-i H t} = (e^{-i a_1  P_1 t / n} e^{-i  a_2 P_2 t / n} \cdots )^n$$

    Where the hamiltonian is $H = P_1 + P_2 + \cdots$

    The units of this evolution are exp(-iHt * pi/2)

    The trotter ordering is determined by the ordering of the terms in the
    QubitPauliOperator.

    Args:
    ----
        operator (QubitPauliOperator): The Hamiltonian to be approximated.
        n_state_qubits (int): The number of qubits in the state register.
        time_slice (Symbol|float): The time slice of the Trotter step.
        state_qreg_str (str, optional): The string of the state qreg.

    """

    def __init__(
        self,
        operator: QubitPauliOperator,
        n_state_qubits: int,
        time_slice: float | Symbol,
        state_qreg_str: str = "q",
    ):
        """Initialise the TrotterPauliExpBox class."""
        time_slice_op: QubitPauliOperator = operator * time_slice  # type: ignore

        circ = RegisterCircuit(self.__class__.__name__)

        state_qreg = circ.add_q_register(state_qreg_str, n_state_qubits)
        qregs = TrotterQReg(state_qreg)

        d = time_slice_op._dict  # type: ignore
        paulis = [PauliTerm(pauli, coeff) for pauli, coeff in d.items()]  # type: ignore
        for p in paulis:
            circ.add_gate(
                PauliExpBox(p.paulis, p.coeff), [qregs.state[i] for i in p.q_inds]
            )

        super().__init__(qregs, circ)

    def symbol_substitution(self, symbol_map: dict[Symbol, float]):
        """Return a new TrotterPauliExpBox with symbols substituted."""
        self._reg_circuit.symbol_substitution(symbol_map)
