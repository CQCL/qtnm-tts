"""PrepareQROM class."""

import numpy as np
from numpy.typing import NDArray
from qtnmtts.circuits.core import RegisterBox, RegisterCircuit, QRegMap
from math import ceil, log2
from pytket.circuit import OpType, Qubit

from qtnmtts.circuits.qrom import QROMBox
from qtnmtts.circuits.rotations import RotationsQROM
from pytket.circuit import QubitRegister
from qtnmtts.circuits.index.method import IndexMethodBase, IndexDefault
from qtnmtts.circuits.core import make_qreg_dataclass


class PrepareQROM(RegisterBox):
    """State preparation with QROM via multiplexed controlled rotations.

    This class inherits from RegisterBox. It is the class for preparing a state
    using a QROM via controlled rotations. The state is prepared using the multiplexor
    method described in https://www.nature.com/articles/s41598-021-85474-1 algorithm 1.
    The state is prepared by a sequences of multiplexed controlled rotations applying
    controlled rotations to a single qubit. The rotations are controlled by the
    data qubits and fire if the data qubit is 1. The rotations must be in the same
    axis. Each data bit in the QROM is a rotation of 1/(2**(1+n))*pi where n is the
    index of the bit in the data. The precision of the rotation increases with the
    index of the bit by a factor of 2. The roations are in the range [0, 4pi).
    where the double rotation is needed to generate negative states.

    See equation (34) in appendix A of https://www.nature.com/articles/s41598-021-85474-1.

    Registers:
        state (QubitRegister): The state register (default - q).
        data (QubitRegister): The data register (default - d).
        work (QubitRegister) - optional: The work register (default - w).
            depending on the index method used.

    Args:
    ----
        index_method (IndexMethodBase): The index method.
        state (NDArray[np.float64]): The state to be prepared.
        n_precision (int): The number of precision qubits.

    """

    def __init__(
        self,
        index_method: IndexMethodBase,
        state: NDArray[np.float64],
        n_precision: int,
    ):
        """Initialise the PrepareQROM."""
        if state.imag.any():
            raise ValueError("input state is not real")

        self._n_qubits = int(ceil(log2(len(state))))

        chunk_data, self._disc_angles = self._get_disc_chunked_data(state, n_precision)

        data_qreg = QubitRegister("d", n_precision)

        # work qubit methods only work for 2 or larger index qubits
        qrom_box = QROMBox(IndexDefault(), {data_qreg: chunk_data[1]})
        qrom_rotations = [RotationsQROM(qrom_box, OpType.Ry, uncompute=True)]

        for dat in chunk_data[2:]:
            qrom_box = QROMBox(index_method, {data_qreg: dat})
            qrom_rotations.append(RotationsQROM(qrom_box, OpType.Ry, uncompute=True))

        # get data and work qubits
        largest_qrom = qrom_rotations[-1].qrom_box
        circ = largest_qrom.initialise_circuit(no_index=True)
        circ.name = f"{self.__repr__()}"

        state_qreg = circ.add_q_register("q", self._n_qubits)

        qreg_dict = {
            key: value
            for key, value in largest_qrom.qreg.__dict__.items()
            if "index" not in key
        }
        qreg_dict["state"] = state_qreg
        qregs = make_qreg_dataclass(qreg_dict, "PrepareQROMQRegs")

        # Build circuit
        circ.Ry(self._disc_angles[0], state_qreg[0])

        # work qubit methods only work for 2 or larger index qubits
        circ = self._add_rotations(1, circ, qrom_rotations[0], state_qreg)

        # add the rest of the rotations using the index method
        # Map the state qubits to the QROM index qubits in the QROM roations
        for i, qrom_rotation in enumerate(qrom_rotations[1:]):
            circ = self._add_rotations(i + 2, circ, qrom_rotation, state_qreg)

        self._precision = qrom_rotations[-1].precision

        super().__init__(qregs, circ)

    @property
    def precision(self):
        """Return the precision of the rotations."""
        return self._precision

    def _add_rotations(
        self,
        i: int,
        circ: RegisterCircuit,
        qrom_rotation: RotationsQROM,
        state_qreg: QubitRegister,
    ) -> RegisterCircuit:
        """Add the QROM rotations to the circuit.

        This function adds the QROM rotations to the circuit. By adding the
        register box to the circuit. The qubit registers are mapped to the
        correct qubit registers in the circuit.

        The state qubits of the circuit are mapped to the index qubits of the
        QROM box used in the QROMRotations primitive. This loops over the
        1 - > n-1 state qubits.

        Args:
        ----
            i (int): The index of the rotation.
            circ (RegisterCircuit): The circuit to add the rotations to.
            qrom_rotation (RotationsQROM): The rotations to add.
            state_qreg (QubitRegister): The state qubit register.
            qreg_map_init (list[QubitRegister]): The initial qubit registers.

        Returns:
        -------
            RegisterCircuit: The circuit with the rotations added.

        """
        circ_qubits: list[Qubit | list[Qubit]] = [
            q
            for q in qrom_rotation.qrom_box.qubits
            if q not in qrom_rotation.qrom_box.qreg.index.to_list()
        ]
        box_qubits = circ_qubits.copy()

        circ_qubits.extend([state_qreg.to_list()[:i], state_qreg[i]])
        box_qubits.extend(
            [qrom_rotation.qreg.index.to_list(), qrom_rotation.qreg.rot[0]]
        )

        qreg_map = QRegMap(box_qubits, circ_qubits)
        circ.add_registerbox(qrom_rotation, qreg_map)
        return circ

    def _get_disc_chunked_data(
        self, state: NDArray[np.float64], n_precision: int
    ) -> tuple[list[list[list[bool]]], NDArray[np.float64]]:
        """Get the discretised chunked data.

        This function takes in the state and the number of precision qubits and
        returns the discretised chunked data and the discretised angles.

        Args:
        ----
            state (NDArray[np.float64]): The state to be prepared.
            n_precision (int): The number of precision qubits.

        Returns:
        -------
            tuple[list[list[list[bool]]], NDArray[np.float64]]: The discretised chunked
            data and the discretised angles.

        """
        missing_amps = 2**self._n_qubits - len(state)
        state = np.hstack((state, np.zeros(missing_amps)))

        self._angles = self._gen_angles(state) / np.pi

        disc_angles = self._discretise_angles(self._angles, n_precision)

        data = [
            self._angle_to_fixed_point_binary(angle, n_precision)
            for angle in disc_angles
        ]
        chunk_data = [
            data[(2**i) - 1 : (2 ** (i + 1)) - 1]
            for i in range(int(ceil(log2(len(data)))))
        ]
        return chunk_data, disc_angles

    def _angle_to_fixed_point_binary(
        self, angle: np.float64, precision: int
    ) -> list[bool]:
        """Convert angle in [0, 4pi) to fixed point binary representation.

        This function takes in a angle in [0, 4) in units of piand converts it to a
        normalises it between [0,1) and convert it to a fixed point binary
        representation.

        Args:
        ----
            angle (np.float64): The angle to be converted.
            precision (int): The number of precision qubits.

        Returns:
        -------
            list[bool]: The fixed point binary representation of the angle.

        """
        if angle < 0:
            angle = 4 + angle
        angle = angle / 4
        fixed_point = round(angle * (2**precision))
        binary = format(int(fixed_point), f"0{precision}b")
        bool_list = [bool(int(bit)) for bit in binary]

        return bool_list

    def _discretise_angles(
        self, angles: NDArray[np.float64], n_data_qubits: int
    ) -> NDArray[np.float64]:
        """Discretise the angles to the nearest multiple of 1/2^(n_data_qubits + 1).

        Args:
        ----
            angles (NDArray[np.float64]): The angles to be discretised.
            n_data_qubits (int): The number of precision qubits.

        Returns:
        -------
            NDArray[np.float64]: The discretised angles.

        """
        multiple = np.array(1 / (2 ** (1 + n_data_qubits)))
        return np.around(angles / multiple) * multiple

    def _gen_angles(self, amp_list: NDArray[np.float64]) -> NDArray[np.float64]:
        """Generate the angles for the rotations.

        This recursive function calculates the angles in the required
        for the statevector multiplexor. Following the procedure in
        https://www.nature.com/articles/s41598-021-85474-1. Algorithm 1.
        The generated angles are in the range [0, 4pi). Because they need
        to be able to capture the full negative states.

        Args:
        ----
            amp_list (NDArray[np.float64]): The amplitudes of the state.

        Returns:
        -------
            NDArray[np.float64]: The angles for the rotations.

        """
        n = round(np.log2(len(amp_list)))
        N = len(amp_list)

        if len(amp_list) != 2**n:
            raise ValueError("please define qubit state on 2^n amplitudes")

        if amp_list.imag.any():
            raise ValueError("input state is not real")

        if not np.isclose(np.linalg.norm(amp_list), 1):
            raise ValueError("input state is not normalized")

        angles = np.array([])
        if len(amp_list) > 1:
            new_dimension = N // 2
            new_amps = np.zeros(new_dimension)
            for k in range(new_dimension):
                new_amps[k] = np.sqrt(amp_list[2 * k] ** 2 + amp_list[2 * k + 1] ** 2)

            # Recursive function call
            inner_angles: NDArray[np.float64] = self._gen_angles(new_amps)

            angles = np.zeros(new_dimension)
            for k in range(new_dimension):
                if new_amps[k] != 0:
                    if amp_list[2 * k] > 0:
                        angles[k] = 2 * np.arcsin(amp_list[2 * k + 1] / new_amps[k])
                    else:
                        angles[k] = 2 * np.pi - 2 * np.arcsin(
                            amp_list[2 * k + 1] / new_amps[k]
                        )
                else:
                    angles[k] = 0

            angles = np.hstack((inner_angles, angles))

        return angles

    @property
    def angles(self):
        """Return the angles."""
        return self._angles

    @property
    def disc_angles(self):
        """Return the discretised angles."""
        return self._disc_angles
