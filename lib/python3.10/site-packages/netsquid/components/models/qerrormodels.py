# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# File: qerrormodels.py
# 
# This file is part of the NetSquid package (https://netsquid.org).
# It is subject to the NetSquid Software End User License Conditions.
# A copy of these conditions can be found in the LICENSE.md file of this package.
# 
# NetSquid Authors
# ================
# 
# NetSquid is being developed within [Quantum Internet division](https://qutech.nl/research-engineering/quantum-internet/) at QuTech.
# QuTech is a collaboration between TNO and the TUDelft.
# 
# Active authors (alphabetical):
# 
# - Tim Coopmans (scientific contributor)
# - Chris Elenbaas (software developer)
# - David Elkouss (scientific supervisor)
# - Rob Knegjens (tech lead, software architect)
# - Iñaki Martin Soroa (software developer)
# - Julio de Oliveira Filho (software architect)
# - Ariana Torres Knoop (HPC contributor)
# - Stephanie Wehner (scientific supervisor)
# 
# Past authors (alphabetical):
# 
# - Axel Dahlberg (scientific contributor)
# - Damian Podareanu (HPC contributor)
# - Walter de Jong (HPC contributor)
# - Loek Nijsten (software developer)
# - Martijn Papendrecht (software developer)
# - Filip Rozpedek (scientific contributor)
# - Matt Skrzypczyk (software contributor)
# - Leon Wubben (software developer)
# 
# The simulation engine of NetSquid depends on the pyDynAA package,
# which is developed at TNO by Julio de Oliveira Filho, Rob Knegjens, Coen van Leeuwen, and Joost Adriaanse.
# 
# Ariana Torres Knoop, Walter de Jong and Damian Podareanu from SURFsara have contributed towards the optimization and parallelization of NetSquid.
# 
# Hana Jirovska and Chris Elenbaas have built Python packages for MacOS.
# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This file uses NumPy style docstrings: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

"""
The quantum error model interface (abstract base class) defined in this module
allows users to specify custom error (e.g. noise or loss) functionality for quantum components.
Aside from the interface, some examples are also defined.

"""

import math
import numpy as np
import netsquid as ns
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits.qformalism import QFormalism, get_qstate_formalism
from netsquid.util import simtools as simtools
from netsquid.qubits.qubit import Qubit
from netsquid.components.models.errormodels import ErrorModel
from netsquid.util.constrainedmap import ValueConstraint, nonnegative_constr
__all__ = [
    "QuantumErrorModel",
    "DepolarNoiseModel",
    "DephaseNoiseModel",
    "T1T2NoiseModel",
    "FibreLossModel"
]


class QuantumErrorModel(ErrorModel):
    """Interface for a callable object that applies errors to qubits.

    """

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Error operation to apply to qubits (to be overridden).

        Parameters
        ----------
        qubits : list of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply error to.
        delta_time : float, optional
            Time qubits have spent on a component [ns].

        """
        # Should be overridden
        raise NotImplementedError

    def compute_model(self, qubits, delta_time=0, **kwargs):
        """Perform the action of the model.

        This is the method called when the object is used as a callable function.

        Parameters
        ----------
        qubits : list of :obj:`~netsquid.qubits.qubit.Qubit` or None
            Qubits to apply error to.
        delta_time : float, optional
            Time qubits have spent on a component [ns].

        Raises
        ------
        TypeError
            If the list `items` contains anything other than :obj:`~netsquid.qubits.qubit.Qubit` or ``None``.

        Notes
        -----
            For the model to work, the \\*\\*kwargs need to include all the
            parameters listed in the attribute *required properties* of the
            model.

        """
        # NOTE: check for Qubit and None since a loss model can set items to None.
        if not all(isinstance(item, Qubit) or item is None for item in qubits):
            raise TypeError("A QuantumErrorModel requires a list of Qubits as input, not {}".format(qubits))
        self.error_operation(qubits=qubits, delta_time=delta_time, **kwargs)

    @staticmethod
    def lose_qubit(qubits, qubit_index, prob_loss=1., rng=None):
        """Helper function to lose a qubit.

        Parameters
        ----------
        qubits : list of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits from which a qubit should be lost.
        qubit_index : int
            Index of the qubit that should be lost.
        prob_loss : float, optional
            Probability with which the qubit is lost, used in case of number state.
        rng : :obj:`numpy.random.RandomState`, optional
            The random number generator to use. Default `simtools.get_random_state()`.

        Notes
        -----
            In the case of a standard qubit item, it is discarded from its shared quantum state (if applicable).

            If the qubit represents a number state (e.g. presence of a photon), the qubit is amplitude dampened
            according to the loss probability.

        """
        qubit = qubits[qubit_index]
        if qubit is None or qubit.qstate is None:
            return
        if qubit.is_number_state:
            # If qubit is a number state, then we want to amplitude dampen
            # towards |0> but not physically lose it.
            qapi.amplitude_dampen(qubit, gamma=prob_loss, prob=1.)
        else:
            if rng is None:
                rng = simtools.get_random_state()
            if math.isclose(prob_loss, 1.) or rng.random_sample() <= prob_loss:
                qapi.discard(qubit)
                qubits[qubit_index] = None

    @classmethod
    def concatenation_class(cls):
        """Type of the object that is returned when concatenating this model

        Returns
        -------
        :class:`~netsquid.components.models.qerrormodels.QuantumErrorModel`
            Concatenating two (subclasses of) QuantumErrorModel results in a QuantumErrorModel.

        """
        return QuantumErrorModel


class DepolarNoiseModel(QuantumErrorModel):
    """Model for applying depolarizing noise to qubit(s) on a quantum component.

    Parameters
    ----------
    depolar_rate : float
        Probability that qubit will depolarize with time. If ``time_independent`` is False (default),
        then this is the exponential depolarizing rate per unit time [Hz].
        If True, it is a probability.
    time_independent : bool, optional
        Whether the probability of depolarizing is time independent. If True,
        it is interpreted as a probability. Default is False.

    """

    def __init__(self, depolar_rate, time_independent=False, **kwargs):
        super().__init__(**kwargs)
        # NOTE time independence should be set *before* the rate
        self.add_property('time_independent', time_independent, value_type=bool)

        def depolar_rate_constraint(value):
            if self.time_independent and not 0 <= value <= 1:
                return False
            elif value < 0:
                return False
            return True
        self.add_property('depolar_rate', depolar_rate,
                          value_type=(int, float),
                          value_constraints=ValueConstraint(depolar_rate_constraint))

    @property
    def depolar_rate(self):
        """float: probability that a qubit will depolarize with time. If
        :attr:`~netsquid.components.models.qerrormodels.DepolarNoiseModel.time_independent`
        is False, then this is the exponential depolarizing rate per unit time [Hz].
        If True, it is a probability."""
        return self.properties['depolar_rate']

    @depolar_rate.setter
    def depolar_rate(self, value):
        self.properties['depolar_rate'] = value

    @property
    def time_independent(self):
        """bool: Whether the probability of depolarizing is time independent."""
        return self.properties['time_independent']

    @time_independent.setter
    def time_independent(self, value):
        self.properties['time_independent'] = value

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Error operation to apply to qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to.
        delta_time : float, optional
            Time qubits have spent on a component [ns]. Only used if
            :attr:`~netsquid.components.models.qerrormodels.DepolarNoiseModel.time_independent` is False.

        """
        if self.time_independent:
            for qubit in qubits:
                if qubit is not None:
                    qapi.depolarize(qubit, prob=self.depolar_rate)
        else:
            for qubit in qubits:
                if qubit is not None:
                    qapi.delay_depolarize(qubit, depolar_rate=self.depolar_rate, delay=delta_time)


class DephaseNoiseModel(QuantumErrorModel):
    """Model for applying dephasing noise to qubit(s) on a quantum component.

    Parameters
    ----------
    dephase_rate : float
        Probability that qubit will dephase with time. If ``time_independent`` is False,
        then this is the exponential depolarizing rate per unit time [Hz].
        If True, it is a probability.
    time_independent : bool, optional
        Whether the probability of depolarizing is time independent. If True,
        it is interpreted as a probability. Default is False.

    """

    def __init__(self, dephase_rate, time_independent=False, **kwargs):
        super().__init__(**kwargs)
        # NOTE time independence should be set *before* the rate
        self.add_property('time_independent', time_independent, value_type=bool)

        def dephase_rate_constraint(value):
            if self.time_independent and not 0 <= value <= 1:
                return False
            elif value < 0:
                return False
            return True
        self.add_property('dephase_rate', dephase_rate,
                          value_type=(int, float),
                          value_constraints=ValueConstraint(dephase_rate_constraint))

    @property
    def dephase_rate(self):
        """float: probability that a qubit will dephase with time. If
        :attr:`~netsquid.components.models.qerrormodels.DephaseNoiseModel.time_independent`
        is False, then this is the exponential depolarizing rate per unit time [Hz].
        If True, it is a probability."""
        return self.properties['dephase_rate']

    @dephase_rate.setter
    def dephase_rate(self, value):
        self.properties['dephase_rate'] = value

    @property
    def time_independent(self):
        """bool: Whether the probability of depolarizing is time independent."""
        return self.properties['time_independent']

    @time_independent.setter
    def time_independent(self, value):
        self.properties['time_independent'] = value

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Error operation to apply to qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to.
        delta_time : float, optional
            Time qubits have spent on a component [ns]. Only used if
            :attr:`~netsquid.components.models.qerrormodels.DepolarNoiseModel.time_independent` is False.

        """
        if self.time_independent:
            for qubit in qubits:
                if qubit is not None:
                    qapi.dephase(qubit, prob=self.dephase_rate)
        else:
            for qubit in qubits:
                if qubit is not None:
                    qapi.delay_dephase(qubit, dephase_rate=self.dephase_rate, delay=delta_time)


class T1T2NoiseModel(QuantumErrorModel):
    """Commonly used phenomenological noise model based on T1 and T2 times.

    Parameters
    ----------
    T1 : float
        T1 time, dictating amplitude damping component.
    T2: float
        T2 time, dictating dephasing component. Note that this is what is called
        T2 Hahn, as opposed to free induction decay T2\\*

    Raises
    ------
    ValueError
        If T1 or T2 are negative, or T2 > T1 when both are greater than zero.

    Notes
    -----
        Implementation and tests imported from the EasySquid project.

    """

    def __init__(self, T1=0, T2=0, **kwargs):
        super().__init__(**kwargs)

        def t1_constraint(t1):
            if t1 < 0:
                return False
            t2 = self.properties.get('T2', 0)
            if t1 == 0 or t2 == 0:
                return True
            if t2 > t1:
                return False
            return True

        def t2_constraint(t2):
            if t2 < 0:
                return False
            t1 = self.properties.get('T1', 0)
            if t1 == 0 or t2 == 0:
                return True
            if t2 > t1:
                return False
            return True

        self.add_property('T1', T1,
                          value_type=(int, float),
                          value_constraints=ValueConstraint(t1_constraint))
        self.add_property('T2', T2,
                          value_type=(int, float),
                          value_constraints=ValueConstraint(t2_constraint))

    @property
    def T1(self):
        """ float: T1 time, dictating amplitude damping component."""
        return self._properties['T1']

    @T1.setter
    def T1(self, value):
        self._properties['T1'] = value

    @property
    def T2(self):
        """float: T2 time, dictating dephasing component. Note that this is what
        is called T2 Hahn, as opposed to free induction decay \\*T2."""
        return self._properties['T2']

    @T2.setter
    def T2(self, value):
        self._properties['T2'] = value

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Error operation to apply to qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to.
        delta_time : float, optional
            Time qubits have spent on component [ns].

        """
        for qubit in qubits:
            self.apply_noise(qubit, delta_time)

    def apply_noise(self, qubit, t):
        """Applies noise to the qubit, depending on its T1 and T2, and the elapsed access time.

        This follows a standard noise model used in experiments, consisting of T1 and T2 times.
        This model consists of applying an amplitude damping noise (depending on T1), followed by
        dephasing noise (depending on both T1 and T2). If T1 is 0, then only dephasing is applied
        depending on T2. If T2 is 0, then only damping is applied. If both T1 and T2 are set to
        zero, then no noise is applied at all.

        Parameters
        ----------
        qubit : :obj:`netsquid.qubits.Qubit`
            Qubit to apply noise to.
        t : float
            Elapsed time to apply noise for.

        """
        # Check whether the memory is empty, if so we do nothing
        if qubit is None:
            return
        # If no T1 and T2 are given, no noise is applied
        if self.T1 == 0 and self.T2 == 0:
            return
        # Get formalism used within netsquid
        formalism = get_qstate_formalism()
        # If the formalism is density matrices, we can apply amplitude
        # damping and will hence make no approximation to the noise.
        # If we are in the stabilizer or ket formalism, we will approximate
        # using Pauli twirl noise
        if formalism not in QFormalism.ensemble_formalisms:
            # If it's just dephasing noise, then we only apply that which falls
            # into all formalisms. If there is an amplitude damping component,
            # then we approximate according to (PRA, 86, 062318)
            if self.T1 == 0:
                # Apply dephasing noise only
                # Compute the dephasing parameter from T1 and T2
                dp = np.exp(-t / self.T2)
                probZ = (1 - dp) / 2
                # Apply dephasing noise using netsquid lib
                self._random_dephasing_noise(qubit, probZ)
            else:
                # Apply approximation to general noise model (se e.g. PRA, 86, 062318)
                # This approximation is obtained by twirling the model below
                # and results in a Pauli channel
                # Compute probabilities of Pauli channel
                if self.T1 > 0:
                    probX = (1 - np.exp(-t / self.T1)) / 4
                else:
                    probX = 0.25
                probY = probX
                if self.T2 > 0:
                    probZ = (1 - np.exp(-t / self.T2)) / 2 - probX
                else:
                    probZ = 0.5 - probX
                probI = 1 - probX - probZ - probY
                # Apply Pauli noise using netsquid library
                self._random_pauli_noise(qubit, probI, probX, probY, probZ)
        else:
            # Apply standard T1 and T2 decoherence model
            # This means we first apply amplitude damping, followed
            # by dephasing noise, if applicable
            if self.T1 > 0:
                # Apply amplitude damping
                # Compute amplitude damping parameter from T1
                probAD = 1 - np.exp(- t / self.T1)
                # print("probAD: {}".format(probAD))  # XXX
                # Apply amplitude damping noise using netsquid library function
                self._random_amplitude_dampen(qubit, probAD)
            if self.T2 > 0:
                # Apply dephasing noise
                # Compute the dephasing parameter from T1 and T2 (e^(-t/T2)/sqrt(1-probAD))
                if self.T1 == 0:
                    dp = np.exp(-t * (1 / self.T2))
                else:
                    dp = np.exp(-t * (1 / self.T2 - 1 / (2 * self.T1)))
                probZ = (1 - dp) / 2
                # Apply dephasing noise using netsquid lib
                self._random_dephasing_noise(qubit, probZ)

    def _random_amplitude_dampen(self, qubit, probAD, cache=True, cache_precision=-1):
        # For now just apply standard netsquid noise, no special DM probabilistic DM treatment
        ns.qubits.qubitapi.amplitude_dampen(qubit, probAD, prob=1, cache=cache, cache_precision=cache_precision)

    def _random_dephasing_noise(self, qubit, probZ):
        self._random_pauli_noise(qubit, 1 - probZ, 0, 0, probZ)

    def _random_pauli_noise(self, qubit, probI, probX, probY, probZ):
        # For now, just apply standard noise.
        ns.qubits.qubitapi.apply_pauli_noise(qubit, (probI, probX, probY, probZ))


class FibreLossModel(QuantumErrorModel):
    """Model for exponential photon loss on fibre optic channels.

    Uses length of transmitting channel to sample an
    exponential loss probability.

    Parameters
    ----------
    p_loss_init : float, optional
        Initial probability of losing a photon once it enters a channel.
        e.g. due to frequency conversion.
    p_loss_length : float, optional
        Photon survival probability per channel length [dB/km].
    rng : :obj:`~numpy.random.RandomState` or None, optional
        Random number generator to use. If ``None`` then
        :obj:`~netsquid.util.simtools.get_random_state` is used.

    """

    def __init__(self, p_loss_init=0.2, p_loss_length=0.25, rng=None):
        super().__init__()
        self.add_property('p_loss_init', p_loss_init,
                          value_constraints=ValueConstraint(lambda x: 0 <= x <= 1))
        self.add_property('p_loss_length', p_loss_length,
                          value_constraints=nonnegative_constr)
        self.add_property('rng', rng, value_type=np.random.RandomState)
        self.rng = rng if rng else simtools.get_random_state()
        self.required_properties = ["length"]

    @property
    def rng(self):
        """ :obj:`~numpy.random.RandomState`: Random number generator."""
        return self.properties['rng']

    @rng.setter
    def rng(self, value):
        self.properties['rng'] = value

    @property
    def p_loss_init(self):
        """float: initial probability of losing a photon when it enters channel."""
        return self.properties['p_loss_init']

    @p_loss_init.setter
    def p_loss_init(self, value):
        self.properties['p_loss_init'] = value

    @property
    def p_loss_length(self):
        """float: photon survival probability per channel length [dB/km]."""
        return self.properties['p_loss_length']

    @p_loss_length.setter
    def p_loss_length(self, value):
        self.properties['p_loss_length'] = value

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Error operation to apply to qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to.
        delta_time : float, optional
            Time qubits have spent on a component [ns].

        """
        # self.apply_loss(qubits, delta_time, **kwargs)
        for idx, qubit in enumerate(qubits):
            if qubit is None:
                continue
            prob_loss = 1 - (1 - self.p_loss_init) * np.power(10, - kwargs['length'] * self.p_loss_length / 10)
            self.lose_qubit(qubits, idx, prob_loss, rng=self.properties['rng'])
