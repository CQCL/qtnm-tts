"""LCUCircBox Concrete class."""

from qtnmtts.circuits.prepare import PrepareBox
from qtnmtts.circuits.select import SelectBox
from qtnmtts.circuits.lcu import LCUBox


class LCUCustomBox(LCUBox):
    """LCUCustomBox Concrete class.

    This class inherits from LCUBox. It sets the prepare box and select box
    to be any pytket circbox. To then be passed to the parent class LCUBox.
    Where the circuit construction logic is done.

    Registers:
        prepare_qreg (QubitRegister): The prepare register (default - p).
        state_qreg (QubitRegister): The state register (default - q).

    Args:
    ----
        prepare_box (PrepareBox): The prepare box to be used.
        select_box (SelectBox): The select box to be used.

    """

    def __init__(self, prepare_box: PrepareBox, select_box: SelectBox) -> None:
        """Initialise the LCUCustomBox."""
        super().__init__(prepare_box, select_box)
