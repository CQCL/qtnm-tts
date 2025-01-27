"""Unary Iteration Box."""

from __future__ import annotations
from qtnmtts.circuits.index.method import IndexMethodBase

from qtnmtts.circuits.core import RegisterCircuit, QRegMap
from pytket.circuit import CircBox, Qubit, Op
from pytket._tket.circuit import Circuit
from pytket.circuit import QubitRegister
from typing import Any, TYPE_CHECKING
from dataclasses import dataclass


if TYPE_CHECKING:
    from qtnmtts.circuits.index import IndexOperations
    from qtnmtts.circuits.index import IndexOpMapRegs


@dataclass
class IndexUnaryItQRegs:
    """Unary Iteration qubit registers."""

    index: QubitRegister
    work: QubitRegister
    target: list[QubitRegister]


class IndexUnaryIteration(IndexMethodBase):
    """Index circuit with unary iteration.

    Indexed operations implemented using the unary iteration method presented in
    https://arxiv.org/abs/1805.03662.

    Note: We no not cancel the gate outside the range as shown in Figure 3.

    Registers:
        index_qreg (QubitRegister): The index qubit register. (default - i)
        work_qreg (QubitRegister): The work qubit register. (default - w)
        target_q_registers (list[QubitRegister]): The target qubit registers.
            default - [t_0, t_1, ... t_n]
    """

    def __init__(self, toffoli: Op | None = None, uncompute_toffoli: Op | None = None):
        """Initialise the IndexPytket.

        The toffoli and uncompute toffoli gates can be provided. If not provided the
        CCX gate is used. The uncompute toffoli gate is the dagger of the toffoli gate.
        if just the toffoli gate is provided the uncompute toffoli gate is set to the
        dagger of the toffoli gate.

        Args:
        ----
            toffoli (CircBox): The toffoli gate to be used in the circuit.
                default - CCX gate.
            uncompute_toffoli (CircBox): The uncompute toffoli gate to be used in the
            circuit. default - CCX gate.dagger.

        """
        if toffoli is None and uncompute_toffoli is None:
            self._toffoli = CircBox(Circuit(3).CCX(0, 1, 2))
            self._uncompute_toffoli: Op = self._toffoli.dagger
        elif toffoli is not None and uncompute_toffoli is None:
            self._toffoli = toffoli
            self._uncompute_toffoli: Op = toffoli.dagger
        elif toffoli is not None and uncompute_toffoli is not None:
            self._toffoli = toffoli
            self._uncompute_toffoli = uncompute_toffoli
        elif toffoli is None and uncompute_toffoli is not None:
            raise ValueError("Provide toffoli as well as uncompute_toffoli")

    def index_circuit(
        self, indexed_ops: IndexOperations, work_qreg_str: str = "w"
    ) -> tuple[RegisterCircuit, Any]:
        """Create the index circuit with unary iteration.

        Iterates through indices with the relevant cascade up and down
        operations, where the maximum upward cascade is calculated from the first
        different bit between bit string i and i+1.

        Args:
        ----
            indexed_ops (IndexOperations): The indexed operations to be applied.
            work_qreg_str (str): The name of the work qubit register.

        Returns:
        -------
            A tuple containing the circuit and the qreg dataclass.

        """
        circ = RegisterCircuit("IndexUnIt")

        if indexed_ops.n_index_qubits < 2:
            raise ValueError("Unary iteration requires at least 2 index qubits")

        self._indexed_ops = indexed_ops

        self._index_qreg = circ.add_q_register(self._indexed_ops.index_qreg)

        self._work_qreg = circ.add_q_register(
            work_qreg_str, self._indexed_ops.n_index_qubits - 1
        )
        for t_qreg in self._indexed_ops.target_q_registers:
            circ.add_q_register(t_qreg)

        self._qregs = IndexUnaryItQRegs(
            self._index_qreg, self._work_qreg, self._indexed_ops.target_q_registers
        )

        # Calculate the different bits between index_bools[i] and index_bools[i-1]
        # we work out the different bits so we know which part of the control condition
        # we need to recompute
        diffs: list[list[bool]] = []
        for i in range(1, self._indexed_ops.n_index):
            diffs.append(
                [
                    a != b
                    for a, b in zip(
                        self._indexed_ops.index_bools[i - 1],
                        self._indexed_ops.index_bools[i],
                        strict=True,
                    )
                ]
            )

        # get the index of the first different bits between index_bools[i]
        # and index_bools[+1] that are used to calculate the cascade up for each index
        self._first_diff = [diff.index(True) for diff in diffs]

        self._bottom_q_ind = self._indexed_ops.n_index_qubits - 1
        self._last_index = self._indexed_ops.n_index - 1

        for i, (operation, bools) in enumerate(
            zip(self._indexed_ops.index, self._indexed_ops.index_bools, strict=False)
        ):
            circ = self._build_index_circ(i, circ, operation, bools)

        return circ, self._qregs

    @property
    def has_work(self) -> bool:
        """Return if the index method has work qubits."""
        return True

    def _qcontrol(
        self, circ: RegisterCircuit, operation: IndexOpMapRegs, control_qubit: Qubit
    ):
        """Add the qcontrol to the circuit.

        The control is applied to the bottom work qubit in the unary
        iteration method. This works across multiple target qubit registers.

        Args:
        ----
            circ (RegisterCircuit): The circuit to add the qcontrol to.
            operation (IndexOpMapRegs): The operation to be applied.
            control_qubit (Qubit): The control qubit.

        """
        for reg_operation in operation.op_map_reg:
            qc_box = reg_operation.box.qcontrol(1)
            qreg_map = QRegMap(
                [qc_box.qreg.control[0], reg_operation.targ_qreg_map.box_qubits],
                [control_qubit, reg_operation.targ_qreg_map.circ_qubits],
            )
            circ.add_registerbox(qc_box, qreg_map)
        return circ

    def _index_toffoli(self, toffoli: Op, control_0: bool, control_1: bool):
        """Create the Indexed Toffoli gate for the index method.

        Take the Toffoli gate apply the X gates to it to generate the
        0, 1 Indexed Toffoli gate.

        Args:
        ----
            toffoli (Circuit): The Toffoli gate.
            control_0 (bool): The first control bool.
            control_1 (bool): The second control bool.

        """
        circ = Circuit(3, f"Toffoli{int(control_0), int(control_1)}")
        if not control_0:
            circ.X(0)
        if not control_1:
            circ.X(1)
        circ.add_gate(toffoli, [0, 1, 2])
        if not control_0:
            circ.X(0)
        if not control_1:
            circ.X(1)
        return CircBox(circ)

    def _build_index_circ(
        self,
        i: int,
        circ: RegisterCircuit,
        operation: IndexOpMapRegs,
        bools: list[bool],
    ):
        """Build the index components for the unary iteration method.

        Each index starts with the adjacent AND which is calculated from the first
        different bit between i and i+1. The first index does not start with a CX.
        See figure 7 from https://arxiv.org/pdf/1805.03662. Apparent from the first
        index which only cascades down.

        Args:
        ----
            i (int): The index of the circuit.
            circ (RegisterCircuit): The circuit to add the index to.
            operation (IndexOpMapRegs): The operation to be applied.
            bools (list[bool]): The bit string for the index.

        Returns:
        -------
            The circuit with the index components added.

        """
        # First index cascade down from the top qubit to the bottom qubit
        if i == 0:
            circ = self._cascade_down(i, circ, bools)
            circ = self._qcontrol(circ, operation, self._work_qreg.to_list()[-1])
        else:
            # add the adjacent AND added fist between the first diff of i and i-1
            circ = self._adjacent_and(i, circ, bools)

            # if first diff is not the bottom qubit then cascade down
            if self._first_diff[i - 1] != self._bottom_q_ind:
                circ = self._cascade_down(i, circ, bools)

            # add the qcontrol to the bottom work qubit
            circ = self._qcontrol(circ, operation, self._work_qreg.to_list()[-1])

            # if the first diff is the bottom qubit then cascade up
            # or if the last index
            if i == self._last_index or self._first_diff[i - 1] == self._bottom_q_ind:
                circ = self._cascade_up(i, circ, bools)

        return circ

    def _cascade_up(self, i: int, circ: RegisterCircuit, bools: list[bool]):
        """Cascade up the work qubits in the unary iteration method.

        The Cascade up finishes at the work qubit of the the first different qubit
        of j+1.

        Args:
        ----
            i (int): The index of the circuit.
            circ (RegisterCircuit): The circuit to add the index to.
            bools (list[bool]): The bit string for the index.

        Returns:
        -------
            The circuit with the cascade up added.

        """
        #
        j = i - 1
        assert j >= 0

        # if not the last index: cascade to the qubit of the first diff of j+1
        if j != self._last_index - 1:
            top_qubit_ind = self._first_diff[j + 1]
        # else if the last index: cascade to the top qubit
        else:
            top_qubit_ind = 0

        # loop over the cascade up
        for q_i in range(self._bottom_q_ind, top_qubit_ind, -1):
            # For n_index there are n_index- 1 work qubits
            if q_i == 1:
                # the top Toffili is acts on index_qreg[0], index_qreg[1], work_qreg[0]
                circ.add_gate(
                    self._index_toffoli(self._uncompute_toffoli, bools[0], bools[1]),
                    [self._index_qreg[0], self._index_qreg[1], self._work_qreg[0]],
                )
            else:
                # the rest is on work_qreg[q-2], index_qreg[q_i], work_qreg[q_i-1]
                circ.add_gate(
                    self._index_toffoli(self._uncompute_toffoli, True, bools[q_i]),
                    [
                        self._work_qreg[q_i - 2],
                        self._index_qreg[q_i],
                        self._work_qreg[q_i - 1],
                    ],
                )
        return circ

    def _cascade_down(self, i: int, circ: RegisterCircuit, bools: list[bool]):
        """Cascade down the work qubits in the unary iteration method.

        The Cascade down starts at the work qubit of the first different qubit of j
        and ends at the bottom work qubit.

        Args:
        ----
            i (int): The index of the circuit.
            circ (RegisterCircuit): The circuit to add the index to.
            bools (list[bool]): The bit string for the index.

        Returns:
        -------
            The circuit with the cascade down added.

        """
        # if not the first index cascade from the first diff of j to the bottom qubit
        if i != 0:
            first_diff_ind = self._first_diff[i - 1]  # One less first diff ind
        # else cascade from the first qubit to the bottom qubit
        else:
            first_diff_ind = 0

        for q_i in range(
            first_diff_ind + 1, self._bottom_q_ind + 1
        ):  # this will run first diff +1 to  self._bottom_q_ind
            # same logic as cascade up
            if q_i == 1:
                circ.add_gate(
                    self._index_toffoli(self._toffoli, bools[0], bools[1]),
                    [self._index_qreg[0], self._index_qreg[1], self._work_qreg[0]],
                )
            else:
                circ.add_gate(
                    self._index_toffoli(self._toffoli, True, bools[q_i]),
                    [
                        self._work_qreg[q_i - 2],
                        self._index_qreg[q_i],
                        self._work_qreg[q_i - 1],
                    ],
                )
        return circ

    def _adjacent_and(self, i: int, circ: RegisterCircuit, bools: list[bool]):
        """Add the adjacent AND to the circuit.

        The adjacent AND is added to the circuit for the unary iteration method.
        The adjacent AND is calculated from the first different bit between i and i+1.
        It is replaces by a CNOT. See figure 7 from https://arxiv.org/pdf/1805.03662.

        Args:
        ----
            i (int): The index of the circuit.
            circ (RegisterCircuit): The circuit to add the index to.
            bools (list[bool]): The bit string for the index.

        Returns:
        -------
            The circuit with the adjacent AND added.

        """
        j = i - 1  # The are 1 less first diff than index bools
        assert j >= 0
        # if the first diff is not 0 or 1 then add a CNOT between work qubits
        if self._first_diff[j] not in [0, 1]:
            circ.CX(
                self._work_qreg[self._first_diff[j] - 2],
                self._work_qreg[self._first_diff[j] - 1],
            )  # always 1 as work qubit
        # else if the first diff is 0 or 1 then add a CNOT between index and work qubits
        elif self._first_diff[j] == 1:
            if bools[0] is False:
                circ.X(self._index_qreg[0])
            circ.CX(self._index_qreg[0], self._work_qreg[0])
            if bools[0] is False:
                circ.X(self._index_qreg[0])
        return circ
