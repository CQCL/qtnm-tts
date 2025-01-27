"""QROM class for binary data storage."""

from __future__ import annotations
from qtnmtts.circuits.index import IndexBox
from qtnmtts.circuits.index.method import IndexMethodBase


from pytket.utils.operators import QubitPauliOperator


class SelectIndexBox(IndexBox):
    """SelectIndexBox class to be used with LCU workflows.

    The `SelectIndexBox` is a register box for doing indexed Pauli operationsin LCU
    using the `IndexBox` abstraction. It is therefore agnostic to the index method
    used. It takes `QubitPauliOperator` as an operator and breaks it into a
    sum of Pauli strings. A `RegisterBox` is formed for each pauli string
    of pytket `Op`s. The phase of the coefficient of the pauli term is contained in
    the pytket `Op` as a `Unitary1Q` gate. This is because the prepare can only
    have positive magnitudes. Hence if this box is used in an LCU the Prepare must
    use the magnitide of the coefficient in polar cords and the phase is absorbed
    into each term in the select. The absorption of the phase and the generation of
    the register boxes is done in the `SerialLCUOperator` class.

    Args:
    ----
        index_method (IndexMethodBase): The index method to be used.
        operator (QubitPauliOperator): The operator to be applied.
        n_state_qubits (int): The number of qubits in the state register.
        index_qreg_str (str): The string to use for the index (default i).

    """

    def __init__(
        self,
        index_method: IndexMethodBase,
        operator: QubitPauliOperator,
        n_state_qubits: int,
        index_qreg_str: str = "i",
    ):
        """Initialise the QROMBox."""
        from qtnmtts.circuits.lcu import SerialLCUOperator

        op_compilation = SerialLCUOperator(operator, n_state_qubits)
        self._is_hermitian = op_compilation.is_hermitian
        op_map_list = op_compilation.op_map_list

        super().__init__(index_method, op_map_list, index_qreg_str)

    @property
    def is_hermitian(self) -> bool:
        """Return True if the operator is hermitian."""
        return self._is_hermitian
