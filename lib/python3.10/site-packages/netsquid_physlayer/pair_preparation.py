#
# Code for state between spin and photon
#

import numpy as np
import abc

import netsquid as ns
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import operators as ops

__all__ = [
    'PairPreparation',
    'ExcitedPairPreparation',
]


class PairPreparation(metaclass=abc.ABCMeta):
    """
    Abstract class for getting the ideal state between communication qubit and photon
    for entanglement generation attempts.
    """

    @abc.abstractmethod
    def generate(self, *args, **kwargs):
        """
        Returns the communication qubit and photon.
        """
        pass


class ExcitedPairPreparation(PairPreparation):
    """
    Prepares states of the form sqrt(alpha)|00> + sqrt(1-a)|11>
    """
    def generate(self, alpha, number_state=True):
        """
        Create two qubits. Note that this does NOT store them anywhere by default, and only
        produces the state.

        :param alpha: float
            State will be sqrt(alpha)|00> + sqrt(1-a)|11>. alpha needs to be in [0,1]
        :param number_state: bool
            If the photon should be a number (presence/absence) state or not.


        Returns
        -------
        [spin, photon]  Two output qubits
        """

        if not ((alpha >= 0) and (alpha <= 1)):
            raise ValueError("Alpha needs to be in the interval [0,1]")

        # Generate spin - photon entanglement
        [spin, photon] = qapi.create_qubits(2)
        photon.is_number_state = number_state

        # Unitary that maps |0> to sqrt(alpha)|1> + sqrt(1-alpha)|0>
        b = np.sqrt(alpha)
        a = np.sqrt(1 - alpha)
        U = ops.Operator("PrepU", np.array([[a, b], [b, -a]]))

        # Turn them into the desired state
        qapi.operate(spin, U)
        qapi.operate([spin, photon], ns.CNOT)

        return [spin, photon]
