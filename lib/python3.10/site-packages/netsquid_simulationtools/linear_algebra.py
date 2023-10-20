from enum import IntEnum
import numpy as np
from netsquid import BellIndex
from scipy.linalg import sqrtm
from netsquid.qubits.ketstates import s0, s1, h0, h1, y0, y1

identity = np.array([[1, 0], [0, 1]])
pauli_z = np.array([[1, 0], [0, -1]])
pauli_x = np.array([[0, 1], [1, 0]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_operators = {0: pauli_x, 1: pauli_y, 2: pauli_z}


class XYZEigenstateIndex(IntEnum):
    r"""Indices for eigenstates of the Pauli X, Y and Z operators.

    Attributes
    ----------
    X0 : int
        The index corresponding to eigenstate of Pauli X with eigenvalue +1,
        :math:`\frac{1}{\sqrt{2}} \left( \vert 0 \rangle + \vert 1 \rangle \right)`
    X1 : int
        The index corresponding to eigenstate of Pauli X with eigenvalue -1,
        :math:`\frac{1}{\sqrt{2}} \left( \vert 0 \rangle - \vert 1 \rangle \right)`
    Y0 : int
        The index corresponding to eigenstate of Pauli Y with eigenvalue +1,
        :math:`\frac{1}{\sqrt{2}} \left( \vert 0 \rangle + i \vert 1 \rangle \right)`
    Y1 : int
        The index corresponding to eigenstate of Pauli Y with eigenvalue -1,
        :math:`\frac{1}{\sqrt{2}} \left( \vert 0 \rangle - i \vert 1 \rangle \right)`
    Z0 : int
        The index corresponding to eigenstate of Pauli Z with eigenvalue +1,
        :math:`\vert 0 \rangle`
    Z1 : int
        The index corresponding to eigenstate of Pauli Z with eigenvalue -1,
        :math:`\vert 1 \rangle`

    """
    X0 = 0
    X1 = 1
    Y0 = 2
    Y1 = 3
    Z0 = 4
    Z1 = 5


xyz_eigenstates = {XYZEigenstateIndex.X0: h0,
                   XYZEigenstateIndex.X1: h1,
                   XYZEigenstateIndex.Y0: y0,
                   XYZEigenstateIndex.Y1: y1,
                   XYZEigenstateIndex.Z0: s0,
                   XYZEigenstateIndex.Z1: s1
                   }


def _convert_to_density_matrix(state):
    if len(state.shape) == 1:
        state = np.array([[element] for element in state])
    if state.shape[1] == 1:
        state = state @ state.conj().T
    return state


def _perform_pauli_correction(state, bell_index):
    """Perform Pauli correction depending on expected Bell state.

    Pauli X, Y or Z is applied to `state` such that if it is the Bell state specified by `bell_index`,
    it is turned into the |Phi+> = (|00> + |11>) / sqrt(2) state.

    The correction is performed on the second qubit in the state.

    Parameters
    ----------
    state : np.array
        Length-4 vector or 4x4 density matrix.
    bell_index : :class:`netsquid.qubits.ketstates.BellIndex`
        Bell index of expected Bell state, i.e. the Bell state that is expected to be the "closest" to `state`.

    Returns
    -------
    np.array
        Corrected state.

    """

    if state.shape == (4,):
        state = np.array([[element] for element in state])

    if bell_index is BellIndex.B01:
        correction = pauli_x
    elif bell_index is BellIndex.B10:
        correction = pauli_z
    elif bell_index is BellIndex.B11:
        correction = 1j * pauli_y  # = X * Z
    else:
        return state

    correction = np.kron(identity, correction)  # embed operator into 2-qubit space

    if state.shape == (4, 1):
        return correction @ state
    elif state.shape == (4, 4):
        return correction @ state @ correction.conj().T


def _fidelity_between_single_qubit_states(state_1, state_2):
    """Calculate the fidelity ("squared" definition) between two single-qubit states.

    Parameters
    ----------
    state_1 : numpy.array
        Can either be a length-2 vector or 2x2 density matrix.
    state_2 : numpy.array
        Can either be a length-2 vector or 2x2 density matrix.

    Returns
    -------
    float
        Fidelity between `state_1` and `state_2`.

    Notes
    -----
    It is also possible to implement this function using NetSquid's fidelity.
    However, this involves assigning qstates to qubits, which is dependent on the qformalism
    (note e.g. that assigning a density matrix to a qubit converts it to a ket vector).
    This function should be independent of the current formalism, and should also not alter the formalism.

    """
    # Netsquid has standard vectors as [[v1], [v2]], while [v1, v2] is easier to work with, so unpack first
    if state_1.shape == (2, 1):
        state_1 = np.array([element[0] for element in state_1.tolist()])
    if state_2.shape == (2, 1):
        state_2 = np.array([element[0] for element in state_2.tolist()])

    if state_1.shape == (2,) and state_2.shape == (2,):  # both ket vector
        overlap = state_1.conj().T @ state_2
        return np.abs(overlap) ** 2
    elif state_1.shape == (2, 2) and state_2.shape == (2,):
        fidelity = state_2.conj().T @ state_1 @ state_2  # <state_2|rho_{state_1}|state_2>
        return np.real(fidelity)
    elif state_1.shape == (2,) and state_2.shape == (2, 2):
        fidelity = state_1.conj().T @ state_2 @ state_1  # <state_1|rho_{state_2}|state_1>
        return np.real(fidelity)
    elif state_1.shape == (2, 2) and state_2.shape == (2, 2):
        fidelity = np.trace(sqrtm(sqrtm(state_1) @ state_2 @ sqrtm(state_1))) ** 2
        return np.real(fidelity)
    else:
        raise TypeError("States do not have correct dimensions.")
