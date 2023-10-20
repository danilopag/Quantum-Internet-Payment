import math

from netsquid.util.simtools import get_random_state
from netsquid.components.models.cerrormodels import ClassicalErrorModel
import netsquid.qubits.qubitapi as qapi
from netsquid.qubits.qubit import Qubit

__all__ = [
    'FixedProbLossModel',
    'LogicalErrorNoiseModel',
]


class FixedProbLossModel(ClassicalErrorModel):
    def __init__(self, prob_loss):
        """
        Parameters
        ----------
        prob_loss : float
            Probability that a single item on the channel is lost.
        """
        # Parent class `ClassicalErrorModel` is an abstract class without
        # an __init__ function, so there is no need to call super().__init__()

        super().__init__()
        if not (0. <= prob_loss <= 1.):
            raise ValueError("The probability of error {} is not in the interval [0, 1]".format(prob_loss))
        self._prob_loss = prob_loss
        self._rng = get_random_state()

    def error_operation(self, items, delta_time=0, **kwargs):
        """Applies the loss model to items modifying the list in place.

        If a lost item is a qubit then its quantum state is modified accordingly.

        Note: this is a copy-and-paste from the deprecated LossModel of NetSquid.

        Parameters
        ----------
        items : list of any or :obj:`~netsquid.qubits.qubit.Qubit`
            Items that may be lost.
        delta_time : float, optional
            Time items have spent on channel [ns].

        Notes
        -----
            In the case of a standard qubit item, if the qubit is computed to be
            lost it is discarded from its shared quantum state (if applicable).

            If the qubit represents a number state (e.g. presence of a photon),
            then the qubit is amplitude dampened according to the loss probability.

        """
        for idx, item in enumerate(items):
            if item is None:
                continue
            is_qubit = isinstance(item, Qubit)
            if is_qubit and item.is_number_state:
                # If qubit is a number state, then we want to amplitude dampen
                # towards |0> but not physically lose it.
                qapi.amplitude_dampen(item, gamma=self._prob_loss, prob=1.)
            elif math.isclose(self._prob_loss, 1.) or self._rng.random_sample() <= self._prob_loss:
                if is_qubit and item.qstate is not None:
                    qapi.discard(item)
                items[idx] = None


class LogicalErrorNoiseModel(ClassicalErrorModel):
    """
    Error model that changes Python object of the following types:
      * int
      * float
      * string
      * tuples of these
      * list of these
    """
    CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    NUM_CHARS = len(CHARS)

    def __init__(self, prob_error):
        """
        Parameters
        ----------
        prob_error : float
            Probability that a single item on the channel suffers from a logical error.
        """
        # Parent class `ClassicalErrorModel` is an abstract class without
        # an __init__ function, so there is no need to call super().__init__()

        super().__init__()
        if not (0. <= prob_error <= 1.):
            raise ValueError("The probability of error {} is not in the interval [0, 1]".format(prob_error))
        self._prob_error = prob_error
        self._rng = get_random_state()  # the random number generator

    @property
    def prob_item_error(self):
        return self._prob_error

    def error_operation(self, items, delta_time=0, **kwargs):
        """Classical noise to apply to an item (to be overriden).

        Parameters
        ----------
        itemlist : list of any. List should only hold a single object.
        delta_time : float, optional
            Time bitstream has spent on channel [ns].

        """
        for i in range(len(items)):
            if self._has_error_occurred():
                items[i] = self._apply_logical_error(items[i])

    def _has_error_occurred(self):
        return self._rng.uniform() < self.prob_item_error

    def _apply_logical_error(self, item):
        item_type = type(item)
        if item_type is int:
            # if item is an integer, we randomly add or subtract 1 to/from it
            item += self._rng.choice([1, -1])
            return item

        elif item_type is float:
            # if item is a float, we add a random number in the interval [-1, 1]
            item += 1. - 2. * self._rng.uniform()
            return item

        elif item_type is str:
            if len(item) == 1:
                # String is a single character;
                # in this case we return a different character
                other_characters = list(LogicalErrorNoiseModel.CHARS)
                other_characters.remove(item)
                return self._rng.choice(other_characters)
            else:
                # String is longer than a single character
                # We randomly change one of its characters
                item = list(item)  # turn string in to a list
                random_index = self._rng.randint(len(item))
                item[random_index] = self._apply_logical_error(item=item[random_index])
                item = "".join(item)  # turn list back into a string again
                return item

        elif item_type is tuple:
            # If item is a tuple, change one of its elements
            item = list(item)  # turn tuple in to a list
            item = self._apply_logical_error(item=item)  # change one element
            item = tuple(item)  # turn list back into a tuple again
            return item

        elif item_type is list:
            # if item is a list, then change one of its elements
            randint = self._rng.randint(len(item))
            item[randint] = self._apply_logical_error(item=item[randint])  # change one element
            return item

        else:
            raise Exception("Logical error model cannot handle item {} of type {}".format(item, item_type))
