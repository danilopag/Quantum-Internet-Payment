"""
Contains the class `ParameterSet`,
a standard format for listing collections of parameter values.
For example, one could have parameter values for NV centers
from 2019, but also values from older experiments, e.g. 2017.
Such parameters can be stored as class variables in
subclasses of `ParameterSet`, e.g. `NVParameterSet2019`
and `NVParameterSet2017`.
"""
import abc
from collections import namedtuple
import numpy as np


Parameter = \
    namedtuple("Parameter",
               ["name",
                "units",
                "type",
                "perfect_value",
                "convert_to_prob_fn",
                "convert_from_prob_fn"])

Parameter.__new__.__defaults__ = (None, None)

Parameter.__doc__ = """\
Container class for a parameter.

Parameters
----------
name : str
    Name of the parameter.
units : str
    Units the parameter should be given in. Examples: `km` or `seconds`.
type
    Type that the parameter should be. Examples: `int` or `float`.
perfect_value : Any
    The value that the parameter would have if it were free of any errors,
    e.g. 0.0 for probabilities of error or `numpy.inf` for coherence times.
    Should be of type `type`.
convert_to_prob_fn : function that takes value of type `type` and returns float.
    Function that converts the parameter value into a probability of no-error.
    Optional.
    Example: if :math:`T_1` is the coherence time, then `convert_to_prob_fn(T_1=100)`
    returns `numpy.exp(-1/100)`.
convert_from_prob_fn : function that takes value of type float and returns `type`.
    The inverse of `convert_to_prob_fn`. Optional.
"""


class ParameterSet(metaclass=abc.ABCMeta):
    """
    Abstract class for a collection of
    :obj:`~netsquid_simulationtools.parameter_set.Parameter` objects.

    Intended usage: subclass of `ParameterSet` and
    add parameters as class variables.

    Example
    -------

    >>> from netsquid_simulationtools.parameter_set import Parameter, ParameterSet
    >>> import numpy as np

    First, we define a class for our hardware (e.g. NV, atomic
    ensembles, ion traps, etc.) We specify which parameters
    need to be given.

    >>> class MyFavouriteHardwareParameterSet(ParameterSet):
    >>>
    >>>     _REQUIRED_PARAMETERS = \
    >>>         [Parameter(name="probability_of_dark_count",
    >>>                    type=float,
    >>>                    perfect_value=0.),
    >>>          Parameter(name="coherence_time_T1",
    >>>                    type=float,
    >>>                    perfect_value=np.inf)]

    Let us now only give a value for the dark count probability,
    but not for the coherence time:

    >>> class MyFavouriteHardware2019Incomplete(MyFavouriteHardwareParameterSet):
    >>>
    >>>     probability_of_dark_count = 0.4

    Now instantiating this class raises an error, since
    `coherence_time_T1` cannot be found.

    >>> MyFavouriteHardware2019Incomplete()
    >>> # raises Exception:
    >>> # [AttributeError("'MyFavouriteHardware2019Incomplete' object has
    >>> # no attribute 'coherence_time_T1'")]

    Let us now also add this coherence time:

    >>> class MyFavouriteHardware2019Complete(MyFavouriteHardwareParameterSet):
    >>>
    >>>     probability_of_dark_count = 0.4
    >>>     coherence_time_T1 = 10.

    Instantiating does not yield errors this time:

    >>> twentynineteen = MyFavouriteHardware2019Complete()

    And we can for example also get the parameters
    in dictionary form:

    >>> print(twentynineteen.to_dict())
    >>> # output:
    >>> # {'probability_of_dark_count': 0.4, 'coherence_time_T1': 10.0}

    Or get a 'perfect' version of the parameters as a dictionary:

    >>> print(twentynineteen.to_perfect_dict())
    >>> # output
    >>> #  {'probability_of_dark_count': 0.0, 'coherence_time_T1': inf}
    """

    _REQUIRED_PARAMETERS = []
    _NOT_SPECIFIED = "Parameter not specified"

    def __init__(self):
        self.verify()

    def verify(self):
        """
        Verify whether all parameters have been
        assigned a value and checks correctness
        of these values. This method is called
        upon initialization of the object of this
        class and needs thus not be called in
        normal circumstances.

        If a value is set to self._NOT_SPECIFIED,
        it will always pass verification.
        """
        exceptions = self._get_verify_errors()
        for exception in exceptions:
            raise exception

    def _get_verify_errors(self):
        exceptions = []
        for parameter in self._REQUIRED_PARAMETERS:
            try:
                value = self._get_value_by_name(name=parameter.name)
                if not isinstance(parameter.name, str):
                    raise TypeError("Name {} is not of type str".format(parameter.name))
                if parameter.perfect_value is not None and \
                        not isinstance(parameter.perfect_value, parameter.type):
                    if parameter.perfect_value is np.inf and parameter.type is int:
                        pass  # "integer infinity"
                    else:
                        raise TypeError("Perfect value {} is not of type {}".format(
                            parameter.perfect_value, parameter.type))
                if not isinstance(value, parameter.type) and value != self._NOT_SPECIFIED:
                    raise TypeError(
                        "Value {} of parameter {} not of type {}".format(
                            value, parameter.name, parameter.type))
            except Exception as exception:
                exceptions.append(exception)
        return exceptions

    def _get_value_by_name(self, name):
        """
        Parameters
        ----------
        name : str

        Returns
        -------
        :obj:`~parameter.Parameter`
        """
        return getattr(self, name)

    def to_dict(self):
        """
        Returns
        -------
        Dict[str: Any]
        """
        return {parameter.name: self._get_value_by_name(name=parameter.name)
                for parameter in self._REQUIRED_PARAMETERS}

    def to_perfect_dict(self):
        """
        Returns
        -------
        Dict[str: Any]
            Dictionary of parameter names and their perfect values.

        Note
        ----
        If a value is set to self._NOT_SPECIFIED, it will be kept
        at that value.
        """
        ret = {}
        for parameter in self._REQUIRED_PARAMETERS:
            if parameter.perfect_value is None or self._get_value_by_name(name=parameter.name) == self._NOT_SPECIFIED:
                # if not perfect value or parameter not specified, keep current value
                val = self._get_value_by_name(name=parameter.name)
            else:
                val = parameter.perfect_value
            ret[parameter.name] = val
        return ret

    @classmethod
    def to_improved_dict(cls, param_dict,
                         param_improvement_dict, improvement_fn,
                         **kwargs):
        """
        Parameters
        ----------
        param_improvement_dict : Dict[str: float]
            Each key should be identical to a key in
            `param_dict`. If equals `None`, then nothing is done.
            The values are the scalar improvement factors `p`
            (see below in examples).

        improvement_fn : function from (Parameter, float, additional arguments
            specified in `**kwargs`) to float.

        Note
        ----
        If a value is set to self._NOT_SPECIFIED, it will be kept
        at that value.

        Example
        -------

        Let us first set up a subclass of `ParameterSet`:

        >>> from netsquid_simulationtools.parameter_set import Parameter, ParameterSet
        >>> import numpy as np
        >>>
        >>>
        >>> class MyFavouriteHardwareParameterSet(ParameterSet):
        >>>
        >>>     _REQUIRED_PARAMETERS = \\
        >>>         [Parameter(name="probability_of_dark_count",
        >>>                    type=float,
        >>>                    perfect_value=0.),
        >>>          Parameter(name="coherence_time_T1",
        >>>                    type=float,
        >>>                    perfect_value=np.inf)]
        >>>
        >>>
        >>> class MyFavouriteHardware2019Complete(MyFavouriteHardwareParameterSet):
        >>>
        >>>     probability_of_dark_count = 0.4
        >>>     coherence_time_T1 = 10.

        Suppose the we want to improve each parameter linearly
        with respect the its perfect value.

        >>> def my_improvement_fn(parameter, current_value, scalar):
        >>>     return current_value + \\
        >>>         scalar * (parameter.perfect_value - current_value)

        >>> parameterset = MyFavouriteHardware2019Complete()
        >>> myvals = parameterset.to_dict()
        >>> print(myvals)
        >>> # output:
        >>> # {'probability_of_dark_count': 0.4, 'coherence_time_T1': 10.0}
        >>>
        >>> improved_myvals = MyFavouriteHardwareParameterSet.to_improved_dict(
        >>>        param_dict=myvals,
        >>>        param_improvement_dict={"probability_of_dark_count": 0.2},
        >>>        improvement_fn=my_improvement_fn)
        >>> print(improved_myvals)
        >>> # output:
        >>> # {'probability_of_dark_count': 0.32, 'coherence_time_T1': 10.0}
        """
        ret = {}
        for name, val in param_dict.items():
            ret[name] = val
        for name, improvement in param_improvement_dict.items():
            parameter = cls._get_parameter_by_name(name=name)
            if ret[name] == cls._NOT_SPECIFIED:
                # if parameter not specified, do not improve
                pass
            else:
                ret[name] = improvement_fn(parameter, ret[name], improvement, **kwargs)
        return ret

    @classmethod
    def _get_parameter_by_name(cls, name):
        for parameter in cls._REQUIRED_PARAMETERS:
            if parameter.name == name:
                return parameter
        return None

    @classmethod
    def parameter_names(cls):
        """
        list of str
            Names of the parameters in this class.
        """
        return [parameter.name for parameter in cls._REQUIRED_PARAMETERS]


def linear_improvement_fn(parameter, current_value, scalar):
    r"""
    Computes a value for a parameter which is improved
    with respect to `curent_value`. This improvement is a linear
    mixture between the current value and the parameter's perfect value.
    To be precise, this function implements

    :math:`(1 - \text{scalar})\cdot \text{current_value}+\text{scalar} \cdot \text{perfect_value}`.

    Can for example be used as input to
    :obj:`netsquid_simulationtools.parameter_set.ParameterSet.to_improved_dict`.

    Parameters
    ----------
    parameter : :obj:`netsquid_simulationtools.parameter_set.ParameterSet.Parameter`
    current_value : float
    scalar : float

    Returns
    -------
    float

    Raises
    ------
    ValueError if `parameter.perfect_value` equals `NumPy.inf`.
    """
    if parameter.perfect_value == np.inf:
        raise ValueError
    return current_value + \
        scalar * (parameter.perfect_value - current_value)


def rootbased_improvement_fn(parameter, current_value, factor):
    r"""
    Computes a value for a parameter which is improved
    with respect to `curent_value`. This improvement is based upon
    improving the the probability of no-error to which the parameter
    value corresponds using the k-th root.
    To be precise, the improved probability of no-error is

    :math:`\left(\text{prob}_{\text{no error}}\right) ^ {1/\text{factor}}`

    where :math:`\text{prob}_{\text{no error}}` is the corresponding
    probability of no-error and is computed using the function
    :obj:`netsquid_simulationtools.parameter_set.Parameter.to_prob_fn`.

    Can for example be used as input to
    :obj:`netsquid_simulationtools.parameter_set.ParameterSet.to_improved_dict`.

    Parameters
    ----------
    parameter : :obj:`netsquid_simulationtools.parameter_set.ParameterSet.Parameter`
    current_value : float
    factor : float

    Returns
    -------
    float

    Raises
    ------
    ValueError if `parameter.perfect_value` equals `NumPy.inf`.
    """
    if parameter.convert_to_prob_fn is not None:
        probability = parameter.convert_to_prob_fn(current_value)
        improved_probability = probability ** (1. / factor)
        return parameter.convert_from_prob_fn(improved_probability)
    return current_value
