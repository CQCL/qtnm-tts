# from qtnmtts.circuits.lcu import LCUMultiplexorBox
from qtnmtts.circuits.core import RegisterBox, RegisterCircuit


from pytket.circuit import QubitRegister, OpType, Qubit

from dataclasses import dataclass


@dataclass
class CSWAPQRegNames:
    c: Qubit
    a: list[QubitRegister]
    b: list[QubitRegister]


class CSWAPRegisterBox(RegisterBox):
    """Abstract CSWAP base class.

    This class inherits from RegisterBox. The circuit consists of a one-qubit
    control register, and two state registers with same number of qubits,
    n_state_q. Each register of the list of registers A is swapped
    with the equivalent paired register in register list B. This RegisterBox is useful
    for computing squared overlaps and antisymmetrisation of states.

    Registers:
        c (QubitRegister): The control qubit with H-Ctrl[box]-H gates.
            (default - c)
        a list(QubitRegister): The state1 register (default - a).
        b list(QubitRegister): The state2 register (default - b).


    Args:
    ----
        control_qubit (Qubit): The control qubit.
        a_qregs (list[QubitRegister]): The state1 registers to be swapped.
        b_qregs (list[QubitRegister]): The state2 registers to be swap.


    """

    def __init__(
        self,
        control_qubit: Qubit,
        a_qregs: list[QubitRegister],
        b_qregs: list[QubitRegister],
        # should we just give the registers and the control qubit
    ):
        if len(a_qregs) != len(b_qregs):
            raise ValueError("The number of state registers must be equal.")

        n_swap_regisers = len(a_qregs)

        circ = RegisterCircuit(self.__class__.__name__)
        circ.add_qubit(control_qubit)
        a_qregs = [circ.add_q_register(qreg) for qreg in a_qregs]
        b_qregs = [circ.add_q_register(qreg) for qreg in b_qregs]

        # c_qreg = QubitRegister(control_qubit.reg_name, 1)

        qregs = CSWAPQRegNames(control_qubit, a_qregs, b_qregs)

        for n in range(n_swap_regisers):
            for a, b in zip(a_qregs[n], b_qregs[n], strict=True):
                circ.add_gate(OpType.CSWAP, [control_qubit, a, b])

        super().__init__(qregs, circ)
