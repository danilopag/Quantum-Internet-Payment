from dataclasses import dataclass
import numpy as np
from scipy.linalg import sqrtm
import logging

from netsquid.components.qdetector import QuantumDetector, QuantumDetectorError
from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits.ketstates import BellIndex
from netsquid.qubits.qubit import Qubit
from netsquid.util.simtools import sim_time
from netsquid.util.simlog import logger

__all__ = [
    'BSMOutcome',
    'QKDOutcome',
    'TwinDetector',
    'BSMDetector',
    'QKDDetector',
    'ModeError',
]


@dataclass
class BSMOutcome:
    """Possible outcomes of a photon-based Bell state measurement (BSM), caused by either
    a successful single-photon measurement, or
    a combination of photon losses and dark counts (see `BSMDetector._set_meas_operators()`).

    Attributes
    ----------
    success : bool
        Denotes whether the measurement was a success. It has value True if exactly one photon was detected in one
        of the detectors, while zero photons were detected at the other one. In case of dual-rail encoding, this must
        occur twice in each mode. In all other cases, the BSM is not successful.
    bell_index : BellIndex
        Denotes the Bell index of the state, which is only relevant in case the BSM is successful. Must be either
        `BellIndex.PSI_PLUS` or `BellIndex.PSI_MINUS`.
        Default value set to -1 to force a user to set the correct BellIndex.
    """
    success: bool
    bell_index: BellIndex = -1


@dataclass
class QKDOutcome:
    """Possible outcomes of a photon measurement that can be used for quantum-key distribution (QKD).
    A measurement can be performed in three bases (X, Y or Z), and
    the measurement outcome can be either 0 or 1.

    Attributes
    ----------
    success : bool
        Denotes whether the measurement was a success. It has value True if exactly one photon was detected in one
        of the detectors, while zero photons were detected at the other one.
    outcome : int in [0, 1]
        Denotes the measurement outcome, which is only relevant in case the measurement is successful. Must be either
        0 or 1. Note that these values are in principle interchangable and completely rely on convention.
    measurement_basis : str in ['X', 'Y', 'Z']
        Measurement basis that is used during the measurement. In the Z basis two photons are measured independently,
        while for the X and Y basis the two incoming photons are interfered using a beam splitter, where
        for the Y basis an additional phase shift is applied to one of the two photons.
    """
    success: bool
    measurement_basis: str
    outcome: int


class TwinDetector(QuantumDetector):
    r"""Component that contains two detectors which serves as a base class for the BSM and QKD detectors.
    Contains the POVMs for setups with and without a beam splitter that interferes the incoming qubits.

    Schematic of setup:

    .. code-block :: text

              Without beam splitter                  With beam splitter (BS)
                ----------------                      --------------------
                |   /|         |                      |   /|             |
        cout0 : > A( |=========< : qin0       cout0 : > A( |===     =====< : qin0
                |   \|         |                      |   \|   \\ //     |
                |              |                      |       ------ BS  |
                |   /|         |                      |   /|   // \\     |
        cout1 : > B( |=========< : qin1       cout1 : > B( |===     =====< : qin1
                |   \|         |                      |   \|             |
                ----------------                      --------------------

    The two detectors are labeled A and B and for the input and output ports the default names
    `qin0`, `qin1`, `cout0` and `cout1` are used.
    Can represent a setup with or without a single beam splitter, based on which measurement operators are set.
    The Hong-Ou-Mandel interference visibility of the beam splitter can be tuned to represent
    for example a delay between the arrival times of two incoming photons.
    The detectors can perform noisy measurements based on the values of the dark count probability and
    (quantum) detection efficiency parameters.
    The two input ports `qin0` and `qin1` handle register incoming qubits,
    after which a (possibly noisy) measurement is done.
    The measurement outcome and any additional information are transmitted using output ports `cout0` and `cout1`.
    Measurements are only possible for single-qubit states in either presence-absence or dual-rail encoding, and
    only performed when the two messages arrive simultaneously.
    The `is_number_state` property of the received qubits is used to determine whether single-rail encoding
    or dual-rail encoding is used.
    The number of multiplexing modes is automatically identified by the number of qubits in the received messages.

    Parameters
    ----------
    name : str
        Name of detector.
    p_dark : float, optional
        Dark-count probability, i.e. probability of measuring a photon while no photon was present, per detector.
    det_eff : float, optional
        Efficiency per detector, i.e. the probability of detecting an incoming photon.
    visibility : float, optional
        Visibility of the Hong-Ou-Mandel dip, also referred to as the photon indistinguishability.
        A visibility of 1 implies perfectly indistinguishable photons arrive at the beam splitter, while
        a visibility of 0 represents completely (classical) distinguishable photons, such that there
        will be no interference.
    num_resolving : bool, optional
        If set to True, photon-number-resolving detectors will be used, otherwise threshold detectors.
    num_input_ports : int, optional
        Number of ports available for qubit input. Must be at least one. Default 1.
    num_output_ports : int, optional
        Number of ports available for measurement outcome output. Must be at least one. Default 1.
    meas_operators : list or tuple of :obj:`~netsquid.qubits.operators.Operator` or None, optional
        Operators used for general single or multi qubit measurements.
        If the number of qubits which has arrived doesn't match the POVM the measurement will fail.
        This means a fail event will be scheduled and the returned message is an empty list.
        If set overrides the observable, otherwise ignored.
    output_meta : dict or None, optional
        Metadata which is added to the output message
    dead_time : float, optional
        Time after the measurement in which the detectors can't be triggered.
        It is possible for qubits to propagate through the system during the dead time.
        Qubits that would arrive at the detectors during their dead_time are discarded when they enter the system.

    """

    def __init__(self, name: str, p_dark: float, det_eff: float, visibility: float, num_resolving: bool,
                 num_input_ports: int, num_output_ports: int, meas_operators: list, output_meta: dict,
                 dead_time: float):
        # Initialize the parameters
        self._p_dark, self._det_eff, self._visibility, self._num_resolving, self._dead_time =\
            None, None, None, None, None
        self.p_dark = p_dark
        self.det_eff = det_eff
        self.visibility = visibility
        self.num_resolving = num_resolving
        self.dead_time = dead_time
        super().__init__(name, num_input_ports=num_input_ports, num_output_ports=num_output_ports,
                         meas_operators=meas_operators, output_meta=output_meta, dead_time=dead_time)

    @property
    def p_dark(self):
        """float : dark-count probability."""
        return self._p_dark

    @p_dark.setter
    def p_dark(self, value):
        """Setter for dark count probability. Measurement operators must be recomputed if this value is changed.

        Parameters
        ----------
        value : float
            New value of the dark count probability.
        """
        if not 1 >= value >= 0:
            raise ValueError(f"Value of dark-count probability p_dark must be in the interval [0,1], not {value}")
        if self._p_dark != value:
            self._parameter_changed = True
        self._p_dark = value

    @property
    def det_eff(self):
        """float : detection efficiency."""
        return self._det_eff

    @det_eff.setter
    def det_eff(self, value):
        """Setter for detection efficiency. Measurement operators must be recomputed if this value is changed.

        Parameters
        ----------
        value : float
            New value of the detection efficiency."""
        if not 1 >= value >= 0:
            raise ValueError(f"Value of detection efficiency det_eff must be in the interval [0,1], not {value}")
        if self._det_eff != value:
            self._parameter_changed = True
        self._det_eff = value

    @property
    def visibility(self):
        """float : Hong-Ou-Mandel interference visibility, which is a measure of the photon indistinguishability."""
        return self._visibility

    @visibility.setter
    def visibility(self, value):
        """Setter for the visibility. Measurement operators must be recomputed if this value is changed.

        Parameters
        ----------
        value : float
            New value of the Hong-Ou-Mandel interference visibility.
        """
        if not 1 >= value >= 0:
            raise ValueError(f"Value of Hong-Ou-Mandel interference visibility must be in the interval [0,1], "
                             f"not {value}")
        if self._visibility != value:
            self._parameter_changed = True
        self._visibility = value

    @property
    def num_resolving(self):
        """bool : indicates whether photon-number-resolving detectors are used."""
        return self._num_resolving

    @num_resolving.setter
    def num_resolving(self, value):
        """Setter for whether the detector is photon-number resolving.
        Measurement operators must be recomputed if this value is changed.

        Parameters
        ----------
        value : bool
            New value of whether photon-number resolving measurement operators are used."""
        if self._num_resolving != value:
            self._parameter_changed = True
        self._num_resolving = value

    @property
    def dead_time(self):
        """float: dead-time of the detector."""
        return self._dead_time

    @dead_time.setter
    def dead_time(self, value):
        """Setter for the detector dead-time.

        Parameters
        ----------
        value : float
            New value of the detector dead-time.
        """
        self._dead_time = value

    def _prob_no_photon_detected(self, n):
        """Probability that no photons are detected at a particular detector, given that n photons arrive.
        This can only occur if all n photons get lost and a dark count does not occur."""
        return (1 - self._p_dark) * (1 - self._det_eff) ** n

    def _prob_exactly_one_photon_detected(self, n):
        """Probability that exactly one photon is detected at a particular detector, given that n photons arrive.
        This can occur if exactly 1 out of n photons gets detected (there exist (n choose 1) = n combinations of this),
        while simultaneously no dark count occurs, or all photons get lost, but a dark count occurs which is registered
        as a single photon."""
        if n == 0:
            # Separate case to circumvent raising 0.0 to a negative power when detection efficiency is 1
            return self._p_dark
        return n * self._det_eff * (1 - self._det_eff) ** (n - 1) * (1 - self._p_dark) + \
            (1 - self._det_eff) ** n * self._p_dark

    def _set_meas_operators_with_beamsplitter(self):
        """Sets the measurement (Kraus) operators for getting a certain click pattern with visibility,
        detector efficiency and dark counts included.
        First we determine the projectors for having a certain number of photons arriving at one of the detectors,
        provided the Hong-Ou-Mandel dip visibility at hand.
        Then we include detection efficiency and dark counts to get a full set of POVMs, after which we find a
        representation in terms of Kraus operators by taking the matrix square root.
        """
        # Start with setting the projective POVMs for a certain number of photons arriving at a detector
        # Assuming mu is real
        mu = np.sqrt(self._visibility)
        # A derivation of these projectors can be found in Appendix D.5 of https://arxiv.org/abs/1903.09778.
        projectors = {(0, 0): np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                      (1, 0): 1 / 2 * np.array([[0, 0, 0, 0], [0, 1, mu, 0], [0, mu, 1, 0], [0, 0, 0, 0]]),
                      (0, 1): 1 / 2 * np.array([[0, 0, 0, 0], [0, 1, -1 * mu, 0], [0, -1 * mu, 1, 0], [0, 0, 0, 0]]),
                      (1, 1): 1 / 2 * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1 - mu * mu]]),
                      (2, 0): 1 / 4 * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1 + mu * mu]]),
                      (0, 2): 1 / 4 * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1 + mu * mu]])}
        # Now include the detector efficiency and dark-count probability to get the resulting measurement outcome POVMs
        # We assume that a dark count *can* occur simultaneously with the detection of a regular photon, based on the
        # detector model described in https://arxiv.org/abs/1109.0194.
        # Furthermore we assume that at most one dark count can occur simultaneously in a single detector.
        # We define the POVMs for photon-number resolving detectors, which can be easily combined in order to construct
        # the POVMs for non-photon-number resolving detectors.
        # Initialize POVMs
        no_photons_at_both = np.zeros([4, 4], dtype=complex)
        one_photon_at_A_none_at_B = np.zeros([4, 4], dtype=complex)
        multiple_photons_at_A_none_at_B = np.zeros([4, 4], dtype=complex)
        one_photon_at_B_none_at_A = np.zeros([4, 4], dtype=complex)
        multiple_photons_at_B_none_at_A = np.zeros([4, 4], dtype=complex)
        at_least_one_photon_at_both = np.zeros([4, 4], dtype=complex)
        # Note that at most two photons in total can arrive simultaneously at a detector
        for m in range(3):
            for n in range(3 - m):
                no_click_m, no_click_n = self._prob_no_photon_detected(m), self._prob_no_photon_detected(n)
                one_photon_m = self._prob_exactly_one_photon_detected(m)
                one_photon_n = self._prob_exactly_one_photon_detected(n)
                # No photons get detected at either detector. This can only occur if
                # m photons arrive at A, which all get lost and no dark count occurs, and
                # n photons arrive at B, which all get lost and no dark count occurs.
                no_photons_at_both += no_click_m * no_click_n * projectors[(m, n)]
                # Exactly one photon gets detected at A, while none are detected at B. This can only occur if
                # m photons arrive at A, of which exactly 1 gets detected, and
                # n photons arrive at B, after which no photons are detected
                one_photon_at_A_none_at_B += one_photon_m * no_click_n * projectors[(m, n)]
                # Multiple photon gets detected at A, while none are detected at B. This can only occur if
                # m photons arrive at A, after which two or more are detected, and
                # n photons arrive at B, after which no photons are detected
                # Here we use the fact that Pr(X >= 2) = 1 - Pr(X = 1) - Pr(X = 0).
                multiple_photons_at_A_none_at_B += (1 - no_click_m - one_photon_m) * no_click_n * projectors[(m, n)]
                # Due to symmetry, we can follow the same reasoning when A and B are switched
                one_photon_at_B_none_at_A += one_photon_n * no_click_m * projectors[(m, n)]
                multiple_photons_at_B_none_at_A += (1 - no_click_n - one_photon_n) * no_click_m * projectors[(m, n)]
                # Finally, we consider the case when at least one photon arrives at both detectors.
                # This can only occur if
                # m photons arrive at detector A, of which at least one gets detected, and
                # n photons arrive at detector B, of which at least one gets detected.
                # Here we use the fact that Pr(X >= 1) = 1 - Pr(X = 0).
                at_least_one_photon_at_both += (1 - no_click_m) * (1 - no_click_n) * projectors[(m, n)]

        # Set the Kraus operator by taking the matrix square root of the POVMs
        if not self.num_resolving:
            # In this case we cannot distinguish between one or two photons getting detected
            n_00 = ops.Operator("n_0", sqrtm(no_photons_at_both))
            n_10 = ops.Operator("n_10", sqrtm(one_photon_at_A_none_at_B + multiple_photons_at_A_none_at_B))
            n_01 = ops.Operator("n_01", sqrtm(one_photon_at_B_none_at_A + multiple_photons_at_B_none_at_A))
            n_11 = ops.Operator("n_11", sqrtm(at_least_one_photon_at_both))
            meas_operators = [n_00, n_10, n_01, n_11]
        else:
            # We have two separate operators for one or two photons arriving at one the detectors, allowing us to
            # identify false positives during entanglement generation
            n_00 = ops.Operator("n_00", sqrtm(no_photons_at_both))
            n_10 = ops.Operator("n_10", sqrtm(one_photon_at_A_none_at_B))
            n_01 = ops.Operator("n_01", sqrtm(one_photon_at_B_none_at_A))
            n_11 = ops.Operator("n_11", sqrtm(at_least_one_photon_at_both))
            n_20 = ops.Operator("n_20", sqrtm(multiple_photons_at_A_none_at_B))
            n_02 = ops.Operator("n_02", sqrtm(multiple_photons_at_B_none_at_A))
            meas_operators = [n_00, n_10, n_01, n_11, n_20, n_02]

        self._meas_operators = meas_operators

    def _set_meas_operators_without_beamsplitter(self):
        """Set the measurement operators for simple measurements in the Z basis.
        Note that we do not distinguish between number and non-number resolving detectors,
        since at most one photon can arrive.
        If a dark count would occur simultaneously with a single photon, we could measure at most
        two photons, but from this we can still deduce that a single regular photon had arrived.
        Note that since we do not have a beam splitter in this case, there is also no Hong-Ou-Mandel
        interference visibility involved."""
        # The projectors without a beam splitter are straight-forward
        projectors = {(0, 0): np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                      (0, 1): np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                      (1, 0): np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]),
                      (1, 1): np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])}
        # Initialize POVMs
        no_clicks_at_both = np.zeros([4, 4], dtype=complex)
        click_at_A_none_at_B = np.zeros([4, 4], dtype=complex)
        click_at_B_none_at_A = np.zeros([4, 4], dtype=complex)
        click_at_both = np.zeros([4, 4], dtype=complex)
        # Note that at most one regular photon can arrive at a detector in this case
        for m in range(2):
            for n in range(2):
                no_click_m = self._prob_no_photon_detected(m)
                no_click_n = self._prob_no_photon_detected(n)
                # No photons get detected at either detector. This can only occur if
                # m photons arrive at A, which all get lost and no dark count occurs, and
                # n photons arrive at B, which all get lost and no dark count occurs.
                no_clicks_at_both += no_click_m * no_click_n * projectors[(m, n)]
                # At least one photon gets detected at A, while none are detected at B. This can only occur if
                # m photons arrive at A, of which exactly 1 gets detected, and
                # n photons arrive at B, after which no photons are detected
                # Here we use the fact that Pr(X >= 1) = 1 - Pr(X = 0)
                click_at_A_none_at_B += (1 - no_click_m) * no_click_n * projectors[(m, n)]
                # Due to symmetry, we can follow the same reasoning when A and B are switched
                click_at_B_none_at_A += no_click_m * (1 - no_click_n) * projectors[(m, n)]
                # Finally, we consider the case when at least one photon arrives at both detectors.
                # This can only occur if
                # m photons arrive at detector A, of which at least one gets detected, and
                # n photons arrive at detector B, of which at least one gets detected.
                # Here we again use the fact that Pr(X >= 1) = 1 - Pr(X = 0).
                click_at_both += (1 - no_click_m) * (1 - no_click_n) * projectors[(m, n)]

        n_00 = ops.Operator("n_00_no_bs", sqrtm(no_clicks_at_both))
        n_01 = ops.Operator("n_01_no_bs", sqrtm(click_at_A_none_at_B))
        n_10 = ops.Operator("n_10_no_bs", sqrtm(click_at_B_none_at_A))
        n_11 = ops.Operator("n_11_no_bs", sqrtm(click_at_both))

        self._meas_operators = [n_00, n_01, n_10, n_11]

    def preprocess_inputs(self):
        raise NotImplementedError("Function must be overriden by subclass")

    def postprocess_outputs(self, dict_port_outcomes):
        raise NotImplementedError("Function must be overriden by subclass")

    @staticmethod
    def _convert_to_mode_encoding(q):
        """Converts a dual-rail encoded single qubit state to a
        two-qubit, presence-absence encoded qubit state in the original modes.
        The state |0> is converted to |01> (no photon in mode 1, one photon in mode 2),
        while the state |1> is converted to |10> (one photon in mode 1, no photon in mode 2).

        Parameters
        ----------
        q : :obj:`~netsquid.qubits.Qubit`
            Incoming dual-rail encoded qubit.

        Returns
        -------
        q_mode_1 : :obj:`~netsquid.qubits.Qubit`
            Single-rail encoded qubit in the first mode.
        q_mode_2 : :obj:`~netsquid.qubits.Qubit`
            Single-rail encoded qubit in the second mode.
        """
        if q is None or q.qstate is None:
            # Create two qubits in state |00>, so no photons in either time window
            q_mode_1, q_mode_2 = qapi.create_qubits(2)
        else:
            q_mode_1 = q
            q_mode_2, = qapi.create_qubits(1)
            # Create entangled state of the two qubits in presence-absence encoding
            qapi.operate([q_mode_1, q_mode_2], ops.CNOT)
            qapi.operate(q_mode_2, ops.X)
        return q_mode_1, q_mode_2

    def _handle_qinput(self, message=None):
        """Override of QuantumDetector method to allow for incoming qubits to be `None`. For the rest left unchanged."""
        # Store or discard qubits when they arrive, and trigger
        meta = message.meta
        sender = meta["rx_port_name"]
        arrival_time = sim_time()
        for qubit in message.items:
            if not isinstance(qubit, Qubit) and qubit is not None:
                raise ValueError(f"A message should contain Qubits, not {qubit}")
            if self.in_dead_time:
                qapi.discard(qubit)
            else:
                self._qubits_per_port[sender].append((arrival_time, qubit, meta))
                if not self.is_triggered:
                    self.trigger()


class BSMDetector(TwinDetector):
    r"""Component that performs a photon-based probabilistic Bell-state measurement (BSM) using linear optics.
    Supports single-rail (presence-absence) and dual-rail encoding for single-qubit states and multiplexing.

    Schematic of setup:

    .. code-block :: text

                --------------------
                |   /|             |
        cout0 : > A( |===     =====< : qin0
                |   \|   \\ //     |
                |       ------ BS  |
                |   /|   // \\     |
        cout1 : > B( |===     =====< : qin1
                |   \|             |
                --------------------

    Contains a single beam splitter (BS), which interferes incoming photons, and two photon detectors A and B.
    Two input ports `qin0` and `qin1` handle register incoming qubits,
    after which a (possibly noisy) measurement is done.
    The measurement outcome, successful mode (if any) and optional additional information are transmitted
    using output ports `cout0` and `cout1`.
    Measurements are only possible for single-qubit states in either presence-absence or dual-rail encoding, and
    only performed when the two messages arrive simultaneously.
    The `is_number_state` property of the received qubits is used to determine whether single-rail encoding
    or dual-rail encoding is used.
    The number of multiplexing modes is automatically identified by the number of qubits in the received messages.

    Parameters
    ----------
    name : str
        Name of detector.
    p_dark : float, optional
        Dark-count probability, i.e. probability of measuring a photon while no photon was present, per detector.
    det_eff : float, optional
        Efficiency per detector, i.e. the probability of detecting an incoming photon.
    visibility : float, optional
        Visibility of the Hong-Ou-Mandel dip, also referred to as the photon indistinguishability.
    num_resolving : bool, optional
        If set to True, photon-number-resolving detectors will be used, otherwise threshold detectors.
    dead_time : float, optional
        Time after the measurement in which the detectors can't be triggered.
        It is possible for qubits to propagate through the system during the dead time.
        Qubits that would arrive at the detectors during their dead_time are discarded when they enter the system.
    allow_multiple_successful_modes : bool, optional
        If set to True, multiple modes can be successful otherwise the detector stops measuring after the first
        successful mode.

    """

    def __init__(self, name, p_dark=0., det_eff=1., visibility=1., num_resolving=False, dead_time=0.,
                 allow_multiple_successful_modes=False):
        super().__init__(name, p_dark=p_dark, det_eff=det_eff, visibility=visibility, num_resolving=num_resolving,
                         num_input_ports=2, num_output_ports=2, meas_operators=[],
                         output_meta={"successful_modes": [None]}, dead_time=dead_time)
        self._allow_multiple_successful_modes = allow_multiple_successful_modes
        # Dictionary that maps an outcome of one of our measurement operators to a BSMOutcome, where
        # 0 : no click
        # 1 : detector A clicked corresponding to |01> + |10> (a Pauli X correction w.r.t |00> + |11>)
        # 2 : detector B clicked corresponding to |01> - |10> (a Pauli Y correction w.r.t |00> + |11>)
        # 3 : both detector A and B clicked
        # Only in case of number resolving
        # 4 : detector A detected 2 or more photons
        # 5 : detector B detected 2 or more photons
        self._measoutcome2bsmoutcome = {0: BSMOutcome(success=False),
                                        1: BSMOutcome(success=True, bell_index=BellIndex.PSI_PLUS),
                                        2: BSMOutcome(success=True, bell_index=BellIndex.PSI_MINUS),
                                        3: BSMOutcome(success=False),
                                        4: BSMOutcome(success=False),
                                        5: BSMOutcome(success=False)}

    def preprocess_inputs(self):
        """Functionality incorporated in `measure()`"""
        pass

    def postprocess_outputs(self, dict_port_outcomes):
        """Functionality incorporated in `measure()`"""
        pass

    def measure(self):
        """Perform a measurement on the received qubits.

        Applies preprocessing to the qubits before the measurement, and
        applies postprocessing to the measured classical outcomes.
        After the measurement all qubits are discarded.

        Override of superclass method to support multiplexed measurements and dual-rail encoded qubit measurements.
        If some of the parameters have changed, the measurement operators are reset.
        First the correct qubit-pair per mode is selected.
        If the qubits are in presence-absence encoding, we can measure the qubits directly,
        but for dual-rail encoding we first transform the qubits to two single-mode qubits
        and perform the double-click scheme.
        The qubit encoding is determined from the `is_number_state` property.
        All the raw measurement outcomes (int) are transformed to a BSMOutcome.
        Additionally, the successful mode (if any) is transmitted as the meta information
        of the classical output message.

        Raises
        ------
        QuantumDetectorError
            If the `is_number_state` property is not the same for the pair of qubits.
        """
        if self._parameter_changed:
            self._set_meas_operators_with_beamsplitter()
            self._parameter_changed = False
        # Get all qubits per port.
        q_lists = [self._qubits_per_port[port_name] for port_name in self._input_port_names]
        arrival_times, qubits, __ = zip(*[item for q_list in q_lists for item in q_list])
        num_modes = len(qubits) // 2
        # Override for multiplexed measurement
        outcomes = [BSMOutcome(success=False)]
        for mode in range(num_modes):
            # Only perform a measurement if the two arrival times are exactly equal
            arrival_time_left, arrival_time_right = arrival_times[mode], arrival_times[num_modes + mode]
            if arrival_time_left != arrival_time_right:
                raise QuantumDetectorError(f"Arrival times of qubits not equal.\nLeft qubit arrived at "
                                           f"{arrival_time_left}, while right qubit arrived at {arrival_time_right}.")
            # Perform pair-wise measurement for a single mode
            qubit_left, qubit_right = qubits[mode], qubits[num_modes + mode]
            # Determine whether the qubits are number states
            # If qubits are number states and they are lost, they are set to `None`
            is_qubit_left_number_state = False if qubit_left is None else qubit_left.is_number_state
            is_qubit_right_number_state = False if qubit_right is None else qubit_right.is_number_state
            if is_qubit_left_number_state is not is_qubit_right_number_state:
                raise QuantumDetectorError(f"BSMDetector {self.name} received a pair of qubits from either side "
                                           "for which one is a number state while the other is not.")
            if is_qubit_left_number_state:
                # Measure in presence-absence encoding
                outcome = qapi.gmeasure([qubit_left, qubit_right], meas_operators=self._meas_operators)[0]
                outcome = self._measoutcome2bsmoutcome[outcome]
            else:
                # Measure in dual-rail encoding in each of the two modes separately
                qubit_left_mode_1, qubit_left_mode_2 = self._convert_to_mode_encoding(qubit_left)
                qubit_right_mode_1, qubit_right_mode_2 = self._convert_to_mode_encoding(qubit_right)
                outcome_mode_1 = qapi.gmeasure([qubit_left_mode_1, qubit_right_mode_1],
                                               meas_operators=self._meas_operators)[0]
                outcome_mode_2 = qapi.gmeasure([qubit_left_mode_2, qubit_right_mode_2],
                                               meas_operators=self._meas_operators)[0]
                outcome_mode_1 = self._measoutcome2bsmoutcome[outcome_mode_1]
                outcome_mode_2 = self._measoutcome2bsmoutcome[outcome_mode_2]
                # A measurement in the double-click scheme is successful if and only if there is a single detector click
                # in both modes
                if outcome_mode_1.success and outcome_mode_2.success:
                    # If the same detector clicked twice the state is presumed to be projected onto
                    # `BellIndex.PSI_PLUS`, otherwise it's `BellIndex.PSI_MINUS`.
                    if outcome_mode_1.bell_index == outcome_mode_2.bell_index:
                        outcome = BSMOutcome(success=True, bell_index=BellIndex.PSI_PLUS)
                    else:
                        outcome = BSMOutcome(success=True, bell_index=BellIndex.PSI_MINUS)
                else:
                    # Not both modes are successful, so the overall outcome is unsuccessful
                    outcome = BSMOutcome(success=False)
            if outcome.success:
                # Append this outcome and the corresponding mode to be transmitted back in a classical message
                # and if multiple modes are not allowed break the for-loop so that the rest of the qubits are
                # not measured (iff not _allow_multiple_successful_modes)
                if not outcomes[0].success:
                    # first success remove the fail placeholder
                    outcomes = []
                outcomes.append(outcome)
                if self._meta["successful_modes"] == [None]:
                    self._meta["successful_modes"] = []
                self._meta["successful_modes"].append(mode)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Successful BSM in BSMDetector {self.name} at time {sim_time()}"
                                 f"with outcome {outcome} and successful mode {mode}.")
                if not self._allow_multiple_successful_modes:
                    break
        # Discard all the qubits
        [qapi.discard(qubit) for qubit in qubits if qubit is not None]
        self._qubits_per_port.clear()
        # Take the measurement outcomes and put the outcomes on the ports
        outcomes_per_port = {port_name: outcomes[:] for port_name in self._output_port_names}
        self.inform(outcomes_per_port)
        # Reset the meta information
        self._meta["successful_modes"] = [None]


class QKDDetector(TwinDetector):
    r"""Detector able to perform photon-based qubit measurements in three bases (X, Y and Z).
     Outcomes can be used to extract a secret key from entangled states used for quantum key distribution (QKD).

    Schematic of setup:

    .. code-block :: text

        Without beam splitter (Z basis)   With beam splitter (BS) and phase shifter (PS)(X and Y bases)
                ----------------                      -----------------------
                |   /|         |                      |   /|           PS   |
                | A( |=========< : qin0               | A( |===     ==|xx|==< : qin0
                |   \|         |                      |   \|   \\ //        |
        cout0 : >              |              cout0 : >       ------ BS     |
                |   /|         |                      |   /|   // \\        |
                | B( |=========< : qin1               | B( |===     ========< : qin1
                |   \|         |                      |   \|                |
                ----------------                      -----------------------

    Able to perform measurements in both single and dual-rail encodings.
    If a list of qubits is received simultaneously on both input ports, single-rail encoding is assumed to be used.
    If a list of qubits is only received on the `qin0` port, dual-rail encoding is assumed to be used.
    The qubits are then converted into two single-rail qubits each.
    Depending on the measurement basis, a different setup is used.
    Measurements in the Z basis are performed by directly measuring the incoming qubits.
    For measurements in the X basis, the incoming qubits are first interfered on a beam splitter.
    For measurements in the Y basis, a phase shift is additionally applied to one of the two qubits.
    The measurement outcome is transmitted using a single classical output port `cout0`.

    Parameters
    ----------
    name : str
        Name of detector.
    measurement_basis : str in ['X', 'Y', 'Z']
        Measurement basis in which the incoming qubits are measured.
    p_dark : float, optional
        Dark-count probability, i.e. probability of measuring a photon while no photon was present, per detector.
    det_eff : float, optional
        Efficiency per detector, i.e. the probability of detecting an incoming photon.
    visibility : float, optional
        Visibility of the Hong-Ou-Mandel dip, also referred to as the photon indistinguishability.
    num_resolving : bool, optional
        If set to True, photon-number-resolving detectors will be used, otherwise threshold detectors.
    dead_time : float, optional
        Time after the measurement in which the detectors can't be triggered.
        It is possible for qubits to propagate through the system during the dead time.
        Qubits that would arrive at the detectors during their dead_time are discarded when they enter the system.

    """
    def __init__(self, name, measurement_basis="Z", p_dark=0., det_eff=1., visibility=1., num_resolving=False, dead_time=0.):
        super().__init__(name, p_dark=p_dark, det_eff=det_eff, visibility=visibility, num_resolving=num_resolving,
                         num_input_ports=2, num_output_ports=1, meas_operators=[ops.Z], output_meta={}, dead_time=dead_time)
        self._measurement_basis = None
        self.measurement_basis = measurement_basis
        self._phase_shifter = ops.Operator("phase_shifter", np.array([[1, 0], [0, 1j]]))

    @property
    def measurement_basis(self):
        """str: the measurement basis used"""
        return self._measurement_basis

    @measurement_basis.setter
    def measurement_basis(self, value):
        """Setter for the measurement basis. If this value is changed, the measurement operators might be changed.

        Parameters
        ----------
        value : str
            New measurement basis to be used.
        """
        if value not in ['X', 'Y', 'Z']:
            raise ValueError(f"Invalid measurement basis, must be either 'X', 'Y' or 'Z' not {value}")
        if self._measurement_basis is not None and self._measurement_basis != value:
            self._parameter_changed = True
        self._measurement_basis = value

    def _measurement2qkdoutcome(self, measurement_outcome):
        if measurement_outcome in [1, 2]:
            return QKDOutcome(success=True, outcome=measurement_outcome - 1, measurement_basis=self._measurement_basis)
        else:
            return QKDOutcome(success=False, outcome=-1, measurement_basis=self._measurement_basis)

    def measure(self):
        """Perform a measurement on the received qubits.

        Applies preprocessing to the qubits before the measurement, and
        applies postprocessing to the measured classical outcomes.
        After the measurement all qubits are discarded.

        Override of superclass method to support multiplexed measurements and dual-rail encoded qubit measurements.
        If some of the parameters have changed, the measurement operators are reset.
        First the correct qubit-pair per mode is selected.
        If the qubits are in presence-absence encoding, we can measure the qubits directly,
        but for dual-rail encoding we first transform the qubits to two single-mode qubits
        and perform the double-click scheme.
        The qubit encoding is determined from the `is_number_state` property.
        All the raw measurement outcomes (int) are transformed to a BSMOutcome.
        Additionally, the successful mode (if any) is transmitted as the meta information
        of the classical output message.

        Raises
        ------
        QuantumDetectorError
            If the `is_number_state` property is not the same for the pair of qubits.
        """
        if self._parameter_changed:
            # Reset the measurement operators based on which basis is used
            if self._measurement_basis == "Z":
                self._set_meas_operators_without_beamsplitter()
            else:
                self._set_meas_operators_with_beamsplitter()
            self._parameter_changed = False

        time_bin = False
        if self._qubits_per_port.get("qin1", None) is None:
            time_bin = True

        # Get all qubits per port.
        q_lists = [self._qubits_per_port[port_name] for port_name in self._input_port_names]
        arrival_times, qubits, metas = zip(*[item for q_list in q_lists for item in q_list])

        num_modes = len(qubits) if time_bin else len(qubits) // 2

        outcomes = []

        for mode in range(num_modes):
            if time_bin:
                # Only a single photon is received on the qin0 port, split it up into two photons and perform rotation
                q_early_or_Left, q_late_or_right = self._convert_to_mode_encoding(qubits[mode])
            else:
                # presence absence
                q_early_or_Left, q_late_or_right = qubits[mode], qubits[num_modes + mode]

                # Only perform a measurement if the two arrival times are exactly equal
                arrival_time_left, arrival_time_right = arrival_times[mode], arrival_times[num_modes + mode]
                if arrival_time_left != arrival_time_right:
                    raise QuantumDetectorError(f"Arrival times of qubits not equal.\nLeft qubit arrived at "
                                               f"{arrival_time_left}, while right qubit arrived at "
                                               f"{arrival_time_right}.")
                # Determine whether the qubits are number states
                # If qubits are number states and they are lost, they are set to `None`
                is_qubit_left_number_state = False if q_early_or_Left is None else q_early_or_Left.is_number_state
                is_qubit_right_number_state = False if q_late_or_right is None else q_late_or_right.is_number_state
                if is_qubit_left_number_state is not is_qubit_right_number_state:
                    raise QuantumDetectorError(f"BSMDetector {self.name} received a pair of qubits from either side "
                                               "for which one is a number states while the other is not.")
                assert is_qubit_left_number_state

            if self._measurement_basis == "Y":
                qapi.operate(q_early_or_Left, self._phase_shifter)

            # measure
            outcome = qapi.gmeasure([q_early_or_Left, q_late_or_right], meas_operators=self._meas_operators)[0]
            outcome = self._measurement2qkdoutcome(outcome)
            outcomes.append(outcome)

        # Discard all the qubits
        [qapi.discard(qubit) for qubit in qubits if qubit is not None]
        self._qubits_per_port.clear()
        # Take the measurement outcomes and put the outcomes on the ports
        outcomes_per_port = {port_name: outcomes[:] for port_name in self._output_port_names}
        self.inform(outcomes_per_port)
        # Reset the meta information
        self._meta["successful_modes"] = None


class ModeError(Exception):
    """Different numbers of modes coming from sources."""
    pass
