"""init file for circuits.utils."""

from .lcu_state_preparation_utils import (
    Rz_jkn,
    extend_functions,
    generate_diagonal_entries,
)
from .lcu_utils import (
    block_encoded_sparse_matrix,
    int_to_bits,
    is_hermitian,
)
from .qsvt_utils import (
    measure_single_qubit_qsp,
    qsp_phase_reflection,
    scipy_qsvt,
    single_qubit_qsp_circuit,
)

from .qsp_angles import (
    ChebyshevPolynomial,
    QSPAngleOptimiser,
    BaseCompilePhases,
    CompilerPhasesNumpy,
    CompilerPhasesNumba,
)

from .block_encoding_utils import generate_diagonal_block_encoding

from .pytket_circboxes import phased_paulig_box

from .linalg import kron_list
from .gqsp_utils import (
    FourierPolynomial,
    GQSPAngleFinder,
)

__all__ = [
    "BaseCompilePhases",
    "ChebyshevPolynomial",
    "CompilerPhasesNumba",
    "CompilerPhasesNumpy",
    "FourierPolynomial",
    "GQSPAngleFinder",
    "QSPAngleOptimiser",
    "Rz_jkn",
    "block_encoded_sparse_matrix",
    "extend_functions",
    "generate_diagonal_block_encoding",
    "generate_diagonal_entries",
    "int_to_bits",
    "is_hermitian",
    "kron_list",
    "measure_single_qubit_qsp",
    "phased_paulig_box",
    "qsp_phase_reflection",
    "scipy_qsvt",
    "single_qubit_qsp_circuit",
]
