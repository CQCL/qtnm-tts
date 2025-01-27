"""Amplitude amplification Oracle Class."""

# from __future__ import annotations
from pytket.circuit import QubitRegister, Qubit
from qtnmtts.circuits.core import RegisterBox, QRegMap
from qtnmtts.circuits.lcu import LCUBox
from qtnmtts.circuits.reflection import ReflectionBox

from dataclasses import dataclass


@dataclass
class AmplificationQRegs:
    """AmplificationBox qubit registers.

    Attributes
    ----------
        prepare (QubitRegister): The prepare register (default - p)
        state (QubitRegister): The state register (default - q)

    """

    prepare: QubitRegister
    state: QubitRegister


class AmplificationBox(RegisterBox):
    """Constructs a AmpAmpBox from an LCUBox.

    Takes an LCUBox box W and does -W R W_dagger R where R is the reflection
    operator kron((2|000...><000...| - 1), Identity). Amplification will be
    effective if the initial success probability is <=0.5.
    If <psi_0| W_dagger W | psi_0> is close to 1, then after one round of
    amplification, new successprobability p_amp is about p*(4*p - 3)**2.
    Otherwise, see Eq. 13 of Berry et.al,  PRL 114, 090502 (2015)

    Registers:
        prepare_qreg (QubitRegister): The prepare register (default - p)
        state_qreg (QubitRegister): The state register (default - q)

    Args:
    ----
        lcu_box (LCUBox): The LCUBox to be used for AmplificationBox.
        iter_num (int): The number of iterations.

    """

    def __init__(self, lcu_box: LCUBox, iter_num: int):
        """Construct an AmplificationBox from an LCUBox."""
        self._lcu_box = lcu_box
        self._reflection_box = ReflectionBox(lcu_box.qreg.prepare.size, positive=False)
        self._iter_num = iter_num

        iter_circ = self._lcu_box.initialise_circuit()

        qreg_map_w = QRegMap(lcu_box.q_registers, iter_circ.q_registers)
        qreg_map_r = QRegMap(
            [self._reflection_box.qreg.reflection], [lcu_box.qreg.prepare]
        )

        # Reflection
        iter_circ.add_registerbox(self._reflection_box, qreg_map_r)

        # W_dagger
        iter_circ.add_registerbox(self.lcu_box.dagger, qreg_map_w)

        # Reflection
        iter_circ.add_registerbox(self._reflection_box, qreg_map_r)

        # W
        iter_circ.add_registerbox(self.lcu_box, qreg_map_w)

        circ = self._lcu_box.initialise_circuit()
        for _ in range(iter_num):
            circ.append(iter_circ.copy())

        circ.name = self.__class__.__name__

        qregs = AmplificationQRegs(
            prepare=lcu_box.qreg.prepare,
            state=lcu_box.qreg.state,
        )
        super().__init__(qregs, circ)

    @property
    def lcu_box(self) -> LCUBox:
        """Return the LCU box."""
        return self._lcu_box

    @property
    def postselect(self) -> dict[Qubit, int]:
        """Return the postselect dictionary."""
        return self.lcu_box.postselect
