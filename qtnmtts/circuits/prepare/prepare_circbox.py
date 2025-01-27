"""PrepareCustomBox class."""

from qtnmtts.circuits.prepare import PrepareBox
from pytket.circuit import Op


class PrepareCircBox(PrepareBox):
    """PrepareCustomBox Concrete class."""

    """This class inherits from PrepareRegisterBox.
    It sets the prepare box to be any pytket circbox.
    To then be passed to the parent class PrepareRegisterBox.
    Where the circuit construction logic is done.

    Registers:
        prepare_qreg (QubitRegister): The prepare register (default - p).

    Args:
        prepare_box (CircBox): The prepare box."""

    def __init__(self, prepare_box: Op, prepare_qreg_str: str = "p") -> None:
        """Initialise the PrepareCustomBox.

        Args:
        ----
            prepare_box (Op): The prepare box to be used.
            prepare_qreg_str (str): The prepare register string. Defaults to "p".

        """
        super().__init__(prepare_box, prepare_qreg_str=prepare_qreg_str)
