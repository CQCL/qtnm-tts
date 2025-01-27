"""QftBox class which implements the standard Quantum Fourier Transform."""

from qtnmtts.circuits.core import RegisterCircuit
from qtnmtts.circuits.core import RegisterBox
from dataclasses import dataclass
from pytket.circuit import QubitRegister


@dataclass
class QFTQRegs:
    """QFTBox qubit registers.

    Attributes
    ----------
        default (QubitRegister): The default register (default - q)

    """

    default: QubitRegister


class QFTBox(RegisterBox):
    """QFT circuit using the standard implementation.

    This QFT implementation incorporates the option to not include the SWAP gates at
    the end of the box. The SWAPs are used to reorder the qubits so the highest
    frequency mode corresponds to the most significant bit. However this operations are
    costly to implement and, since the reordering is known, they can be avoided if this
    change is taken into account for the rest of the circuit and post-processing.

    Registers:
        default (QubitRegister): The default "q" register.

    Args:
    ----
        n_qubits (int): The number of qubits used in the QFT circuit.
        do_swaps (bool): Whether to add swaps or not. Default True.
        default_qreg_str (str): The default qubit register string. Defaults to "q".

    """

    def __init__(
        self,
        n_qubits: int,
        do_swaps: bool = True,
        default_qreg_str: str = "q",
    ):
        """Initialise a QftBox for a given number of qubits."""
        qft_circ = RegisterCircuit()
        qft_circ.name = "QFT"
        self._has_swaps = do_swaps
        qreg = qft_circ.add_q_register(default_qreg_str, n_qubits)
        qregs = QFTQRegs(qreg)

        for i in range(n_qubits):
            qft_circ.H(qreg[i])
            for j in range(i + 1, n_qubits):
                qft_circ.CU1(1 / 2 ** (j - i), qreg[j], qreg[i])

        if do_swaps:
            for k in range(0, n_qubits // 2):
                qft_circ.SWAP(qreg[k], qreg[n_qubits - k - 1])

        self._qreg = qreg

        super().__init__(qreg=qregs, reg_circuit=qft_circ)

    @property
    def has_swaps(self) -> bool:
        """Return if the box has SWAP gates or not."""
        return self._has_swaps
