from netsquid.nodes.connections import DirectConnection
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.models.delaymodels import FibreDelayModel

__all__ = [
    'ClassicalConnection',
]


class ClassicalConnection(DirectConnection):
    def __init__(self, name, delay=0.):
        """A classical connection with a fixed delay.

        Parameters
        ----------
        name : str
            Name of connection for identification purposes.
        delay : float
            The delay in nanoseconds

        """
        channel_AtoB = ClassicalChannel(f"{name}_AtoB", delay=delay)
        channel_BtoA = ClassicalChannel(f"{name}_BtoA", delay=delay)
        super().__init__(name=name, channel_AtoB=channel_AtoB, channel_BtoA=channel_BtoA)


class ClassicalConnectionWithLength(DirectConnection):
    """A classical connection of which the delay is determined by the length and the transmission speed.

    Parameters
    ----------
    name : str
        Name of connection for identification purposes.
    length : float
        Length [km] of the channel.
    c : float, optional
        Speed of transmission [km/s] through the channel.
        In case the connection represents optical fiber, this is the speed of light within that fiber.

    """
    def __init__(self, name, length, c=200000):
        delay_model = FibreDelayModel(c=c)
        channel_AtoB = ClassicalChannel(name=f"{name}_AtoB", models={"delay_model": delay_model}, length=length)
        channel_BtoA = ClassicalChannel(name=f"{name}_BtoA", models={"delay_model": delay_model}, length=length)
        super().__init__(name=name, channel_AtoB=channel_AtoB, channel_BtoA=channel_BtoA)
