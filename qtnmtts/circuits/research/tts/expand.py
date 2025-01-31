"""Module for truncated Taylor Series."""

from pytket.utils.operators import QubitPauliOperator
from pytket._tket.pauli import QubitPauliString
import numpy as np


class ExpQubitOper:
    """Uses Taylor expansion."""

    def __init__(self, operator: QubitPauliOperator, k: np.int64):
        """Initialise the Qubit Pauli Operator and the max order of truncation."""
        self._operator = operator  # operator is the argument of exponential
        self._k = k  # max order of expansion

    def taylor_expand(self) -> QubitPauliOperator:
        """Return the exponential of i times the operator."""
        exp_Op = QubitPauliOperator({QubitPauliString(): 1})
        H_k = QubitPauliOperator({QubitPauliString(): 1})
        for kk in range(self._k):
            # print(kk)
            H_k = H_k * self._operator  # type: ignore
            exp_Op += (1 / np.math.factorial(kk + 1)) * H_k  # type: ignore

        # exp_Op.map(complex)
        exp_Op.compress(10**-self._k)  # type: ignore

        return exp_Op  # type: ignore
