"""SelectCustomBox class."""

from pytket.circuit import CircBox
from qtnmtts.circuits.select import SelectBox


class SelectCircBox(SelectBox):
    """SelectCustomBox Concrete class."""

    """This class inherits from SelectBox.
    It sets the select box to be any pytket circbox.
    To then be passed to the parent class SelectBox.
    Where the circuit construction logic is done.

    Registers:
        prepare_qreg (QubitRegister): The prepare register (default - p).
        state_qreg (QubitRegister): The state register (default - q).


    Args:
        select_box (CircBox): The select box to be used.
        n_state_qubits (int): The number of state qubits.
        prepare_qreg_str (str): The prepare register string. Defaults to "p".
        state_qreg_str (str): The state register string. Defaults to "q".
    """

    def __init__(
        self,
        select_box: CircBox,
        n_state_qubits: int,
        prepare_qreg_str: str = "p",
        state_qreg_str: str = "q",
    ) -> None:
        """Initialise the SelectCustomBox."""
        self._n_prep_qubits = select_box.n_qubits - n_state_qubits

        super().__init__(
            select_box,
            n_state_qubits,
            prepare_qreg_str=prepare_qreg_str,
            state_qreg_str=state_qreg_str,
        )

    @property
    def n_prep_qubits(self) -> int:
        """Return the number of prepare qubits."""
        return self._n_prep_qubits

    @property
    def operator(self):
        """Return the operator."""
        return NotImplementedError("operator cannot be defined for SelectCustomBox")
