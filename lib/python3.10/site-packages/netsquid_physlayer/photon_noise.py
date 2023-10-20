import abc

from netsquid.components.models.qnoisemodels import T1T2NoiseModel

__all__ = [
    'PhotonAbsorptionNoiseModel',
    'PhotonEmissionNoiseModel',
]


class PhotonAbsorptionNoiseModel(T1T2NoiseModel, metaclass=abc.ABCMeta):
    """
    Abstract class for noise model of photon emission.
    Should be subclassed.
    """
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def noise_operation(self, photon, delta_time=0, operator=None, **kwargs):
        """
        Should be subclassed
        """
        pass

    def apply_noise(self, spin, photon, memQubits=[], **kwargs):
        """
        Applies the noise operation to the spin and photon
        and possible other qubits in the memory.
        """
        self.noise_operation([spin, photon] + memQubits, **kwargs)


class PhotonEmissionNoiseModel(T1T2NoiseModel, metaclass=abc.ABCMeta):
    """
    Abstract class for noise model of photon emission.
    Should be subclassed.
    """

    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def noise_operation(self, qubits, delta_time=0, operator=None, **kwargs):
        """
        Should be subclassed.
        This method is expected to return the photon. This is such that the photon can be set to None if its lost.

        Parameters
        ----------
        qubits : list
            [spin, photon] + memory_qubits

        Returns
        -------
        :class:`netsquid.qubits.qubit.Qubit`
            The photon
        """
        pass

    def apply_noise(self, spin, photon, memQubits=[], **kwargs):
        """
        Applies the noise operation to the spin and photon
        and possible other qubits in the memory.
        """
        return self.noise_operation([spin, photon] + memQubits, **kwargs)
