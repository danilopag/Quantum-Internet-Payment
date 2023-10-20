# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# File: qubitapi.py
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

"""General API for manipulating qubits indepedent of their quantum state
*formalism*.

Examples
--------
Several of the API functions provide example code snippets.
These snippets should run unedited if you set up a Python
interpreter with the following imports:

>>> import numpy as np
>>> from netsquid.qubits.qubitapi import *
>>> from netsquid.qubits.qformalism import *
>>> from netsquid.qubits import operators as ops
>>> from netsquid.qubits import ketstates
>>> from netsquid.qubits import Stabilizer

"""
import math
import numpy as np
from netsquid.qubits.kettools import KetRepr
from netsquid.qubits.dmutil import calc_amplitude_dampen_ops
from netsquid.qubits.stabtools import StabRepr, Stabilizer
from netsquid.qubits.dmtools import DenseDMRepr
from netsquid.qubits.qubit import Qubit
from netsquid.qubits.qstate import QStateCombineError, QState
from netsquid.qubits.qrepr import QRepr, convert_to
from netsquid.qubits.state_sampler import StateSampler
from netsquid.qubits import operators as ops
from netsquid.qubits.qformalism import get_qstate_formalism, convert_qstate
from netsquid.util import cymath, simtools
__all__ = [
    "create_qubits",
    "assign_qstate",
    "operate",
    "stochastic_operate",
    "multi_operate",
    "measure",
    "gmeasure",
    "discard",
    "reduced_dm",
    "apply_pauli_noise",
    "depolarize",
    "dephase",
    "amplitude_dampen",
    "apply_dda_noise",
    "delay_depolarize",
    "delay_dephase",
    "fidelity",
    "exp_value",
    "combine_qubits",
]

# Counter for unique default system_name for created qubits
_system_name_counter = 0


class MissingQRepr(Exception):  # DEPRECATED
    """Error associated with QStates that do not have a QRepr object assigned."""
    pass


def _to_qubits_list(qubits, combine=False) -> list:
    # Convert the given qubits into a list of combined qubits
    if isinstance(qubits, Qubit):
        qubits = [qubits]
    elif not isinstance(qubits, list):
        qubits = list(qubits)
    if len(qubits) == 0:
        raise TypeError
    if combine and len(qubits) > 1:
        combine_qubits(qubits)
    return qubits


def _qrepr(qubits) -> QRepr:
    # Retrieve the representation of the qubits
    if isinstance(qubits, Qubit):
        return qubits.qstate.qrepr
    elif isinstance(qubits, list):
        qrepr = qubits[0].qstate.qrepr
        if qrepr is None:  # DEPRECATED
            raise MissingQRepr("This qubit doesn't have a QRepr yet.")
        return qubits[0].qstate.qrepr
    else:
        raise TypeError("The qubits must be given as a list or a single Qubit.")


def _idx(qubits) -> list:
    # This helper function is always executed together with _qrepr, so we skip safety checks
    # and combining.
    if isinstance(qubits, Qubit):
        return qubits.qstate.indices_of([qubits])
    elif isinstance(qubits, list):
        return qubits[0].qstate.indices_of(qubits)
    else:
        raise TypeError("The qubits must be given as a list or a single Qubit.")


def create_qubits(num_qubits, system_name=None, no_state=False):
    r"""Creates a system of qubits.

    By default each qubit is assigned its own independent :math:`\vert 0\rangle` quantum state
    using the currently set formalism.

    Parameters
    ----------
    num_qubits : int
        Number of qubits to create.
    system_name : str or None, optional
        Name to distinguish these qubits from others they will encounter.
        If ``None``, the default system name ``QS#<i>-`` is chosen,
        where ``<i>`` is an incrementing number.
    no_state : bool, optional
        If True the created qubits are not assigned a quantum state, which
        can be assigned later using :func:`~netsquid.qubits.qubitapi.assign_qstate`.
        Default is False.


    Returns
    -------
    list of :obj:`~netsquid.qubits.qubit.Qubit`
        The created qubits.

    Notes
    -----
        The name of each qubit is its ``system_name`` appended with
        an index (starting from 0).

    Example
    -------

    >>> q1, q2, q3 = create_qubits(num_qubits=3, system_name="Q")
    >>> q1.name, q2.name, q3.name
    ('Q0', 'Q1', 'Q2')
    >>> print(q1.qstate.qrepr.reduced_dm())
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]
    >>> print(q2.qstate.qrepr.reduced_dm())
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]
    >>> print(q3.qstate.qrepr.reduced_dm())
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]
    >>> print(reduced_dm([q1, q2]))
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]


    If no ``system_name`` is specified a default name is chosen:

    >>> q1, q2 = create_qubits(2)
    >>> q3, = create_qubits(1)
    >>> q1.name, q2.name, q3.name  # doctest: +SKIP
    ('QS#0-0', 'QS#0-1', 'QS#1-0')

    """
    if system_name is None:
        global _system_name_counter
        system_name = f"QS#{_system_name_counter}-"
        _system_name_counter += 1
    qubits = [Qubit(system_name + str(i)) for i in range(num_qubits)]
    # create quantum states:
    if not no_state:
        # Create qubits in |0> states
        for i in range(num_qubits):
            qrepr_class = get_qstate_formalism()
            qrepr = qrepr_class(num_qubits=1)
            QState(qubits[i:i + 1], qrepr)
    return qubits


def assign_qstate(qubits, qrepr, formalism=None):
    r"""Assign a specific quantum state to qubits.

    Qubits will be discarded from their current shared quantum states if applicable.

    Parameters
    ----------
    qubits : list of :class:`~netsquid.qubits.qubit.Qubit`
        Qubits to assign the quantum state to.
    qrepr : :class:`~netsquid.qubits.qrepr.QRepr`, :class:`numpy.array`, :class:`~netsquid.qubits.stabtools.Stabilizer`,
            :class:`~netsquid.qubits.state_sampler.StateSampler`, None
        Representation of quantum state to assign to the created qubits.
        If an array, it can be a ket vector or a density matrix with a size matching the number of qubits.
        The given representation will be converted into the specified :class:`~netsquid.qubits.qrepr.QRepr`,
        or raise an exception if this is not possible.
    formalism : :class:`~netsquid.qubits.qrepr.QRepr` or None, optional
        Formalism to use for the assigned quantum state. If None (default), the currently set formalism
        is used (recommended).

    Returns
    -------
    :class:`~netsquid.qubits.qstate.QState`
        The shared quantum state assigned to the qubits.

    Raises
    ------
    ValueError
        If ``qrepr`` is in an invalid format.
    NotImplementedError
        If the required quantum state conversion has not been implemented.

    Notes
    -----
        If a mixed state is provided for a pure state formalism, a pure state is
        sampled from it.

    Examples
    --------

    Starting with two qubits (each in ``|0>``) we wish to assign them a new shared state
    e.g. the maximally mixed state represented using a density matrix.
    If we are in the KET formalism, then a pure state will be randomly sampled from
    the specified density matrix (using an eigendecomposition).

    >>> from netsquid.qubits import set_qstate_formalism, QFormalism
    >>> set_qstate_formalism(QFormalism.KET)
    >>> q1, q2 = create_qubits(2)
    >>> print(q1.qstate.qrepr)
    KetRepr(num_qubits=1,
    ket=
    [[1.+0.j]
     [0.+0.j]])

    >>> assign_qstate([q1, q2], np.diag([0.25, 0.25, 0.25, 0.25]))
    >>> print(q1.qstate.qrepr)  # doctest: +SKIP
    KetRepr(num_qubits=2,
    ket=
    [[0.+0.j]
     [1.+0.j]
     [0.+0.j]
     [0.+0.j]]

    >>> print(q1.qstate == q2.qstate)
    True

    To assign a quantum state to new qubits, the ``no_state`` parameter of
    :func:`~netsquid.qubits.qubitapi.create_qubits` can be used.

    >>> set_qstate_formalism(QFormalism.DM)
    >>> q1, q2 = create_qubits(2, no_state=True)
    >>> print(q1.qstate)
    None

    >>> from netsquid.qubits import StabRepr
    >>> stab_repr = StabRepr([[1, 1, 0, 0], [0, 0, 1, 1]], [1, 1])
    >>> assign_qstate([q1, q2], stab_repr)
    >>> print(reduced_dm([q1, q2]))
    [[0.5+0.j 0. +0.j 0. +0.j 0.5+0.j]
     [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
     [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
     [0.5+0.j 0. +0.j 0. +0.j 0.5+0.j]]

    >>> print(q1.qstate == q2.qstate)
    True

    """
    qubits = _to_qubits_list(qubits)
    if formalism is None:
        formalism = get_qstate_formalism()
    if qrepr is None:
        for qubit in qubits:
            discard(qubit)
        return None
    elif isinstance(qrepr, StateSampler):
        qrepr = qrepr.sample().state
    elif not isinstance(qrepr, QRepr):
        # TODO: Remove this deprecation
        if isinstance(qrepr, np.ndarray):
            if len(qrepr.shape) != 2:
                qrepr = qrepr.reshape((-1, 1))
            if np.shape(qrepr)[0] == np.shape(qrepr)[1]:
                qrepr = DenseDMRepr(qrepr)
            elif np.shape(qrepr)[1] == 1:
                qrepr = KetRepr(qrepr)
        elif isinstance(qrepr, Stabilizer):
            qrepr = StabRepr(qrepr.check_matrix, qrepr.phases)
        else:
            raise TypeError("Unknown input.")
    if qrepr.num_qubits != len(qubits):
        raise ValueError(f"The number of qubits given ({len(qubits)}) doesn't match the "
                         f"number of qubits in the representation ({qrepr.num_qubits}).")
    qrepr = convert_to(qrepr, formalism)
    qstate = QState(qubits, qrepr)
    return qstate


def combine_qubits(qubits):
    """Join a list of qubits together so that they share the same quantum state.

    This procedure will generally be performed automatically.
    The order that qubits are internally represented by the resulting
    shared quantum state may differ from the order of the ``qubits`` list.

    Parameters
    ----------
    qubits : list of :obj:`~netsquid.qubits.qubit.Qubit`
        List of qubit to combine.

    Returns
    -------
    list of :obj:`~netsquid.qubits.qubit.Qubit`
        The combined qubits ordered as they are in their new shared state.

    Notes
    -----
        If any qubits already share the same quantum states then their
        combination is ignored.

    Example
    -------
    >>> q1, = create_qubits(num_qubits=1)
    >>> q2, = create_qubits(num_qubits=1)
    >>> q3, = create_qubits(num_qubits=1)
    >>> combine_qubits([q1, q2, q3])
    >>> print(q1.qstate.qrepr.reduced_dm())
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]]

    """
    if None in qubits:
        raise ValueError(f"{qubits} is not a valid list of qubit(s)")
    if len(qubits) < 2:
        return qubits
    qubit1 = qubits[0]
    for qubit2 in qubits[1:]:
        try:
            qubit1.combine(qubit2)
        except QStateCombineError:
            convert_qstate(qubit1)
            convert_qstate(qubit2)
            # Combine again -- should succeed this time
            qubit1.combine(qubit2)
    # Return qubits with the order in which they are stored in the qstate
    return qubit1.qstate.qubits[:]


def measure(qubit, observable=ops.Z, keep_combined=False, discard=False):
    """Projectively measure a qubit.

    The observable operator should be a Hermitian operator.

    Parameters
    ----------
    qubit : :obj:`~netsquid.qubits.qubit.Qubit`
        Qubit to measure (within its quantum state).
    observable : :obj:`~netsquid.qubits.operators.Operator`, optional
        Hermitian operator to measure qubit with. Default is the
        ``Z`` observable.
    keep_combined : bool, optional
        Whether to keep this qubit in its shared quantum state after
        the measurement instead of splitting it into a new state. Not
        applicable if ``discard`` is set to True.
    discard : bool, optional
        Whether to discard the qubit after measurement. If True the quantum
        state of the measured qubit will be ``None``, which is more efficient
        than assigning a new single qubit state.

    Returns
    -------
    int
        0 if positive eigenstate measured, otherwise 1.
    float
        Probability with which this result occurred.

    Raises
    ------
    ValueError
        If ``qubit.qstate`` is None.

    Examples
    --------

    Measure the second qubit from the ``|00>`` state in the
    ``X`` basis:

    >>> q1, q2 = create_qubits(2)  # separate states |0>, |0>
    >>> combine_qubits([q1, q2])  # state |00>
    >>> measure(q2, observable=ops.X) # doctest: +SKIP
    (0, 0.5)
    >>> print(reduced_dm(q2))  # doctest: +SKIP
    [[0.5+0.j 0.5+0.j]
     [0.5+0.j 0.5+0.j]]
    >>> print(reduced_dm(q1))  # doctest: +SKIP
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    Do the same as before, but keep the quantum states combined:

    >>> q1, q2 = create_qubits(2)  # separate states |0>, |0>
    >>> combine_qubits([q1, q2])  # state |00>
    >>> measure(q2, observable=ops.X, keep_combined=True)  # doctest: +SKIP
    (0, 0.499...)
    >>> print(reduced_dm([q1, q2]))  # doctest: +SKIP
    [[0.5+0.j 0.5+0.j  0. +0.j  0. +0.j]
     [0.5+0.j 0.5+0.j  0. +0.j  0. +0.j]
     [0. +0.j 0. +0.j  0. +0.j  0. +0.j]
     [0. +0.j 0. +0.j  0. +0.j  0. +0.j]]

    Note that the ``>>`` operator of :obj:`~netsquid.qubits.operators.Operator`
    has been overloaded to give a shorthand alternative for performing observable
    measurements given a suitable Hermitian operator:

    >>> q1, = create_qubits(1)
    >>> measure(q1, ops.Z)
    (0, 1.0)
    >>> measure(q1, ops.X)  # doctest: +SKIP
    (0, 0.499...)

    """
    # If discarding then always drop the qubit (i.e. do not keep state combined)
    if discard:
        drop_qubit = True
    else:
        drop_qubit = not keep_combined
    if not isinstance(qubit, Qubit):
        raise TypeError("The qubit given must be a qubit object")
    if qubit.qstate is None:
        raise ValueError("Cannot measure a qstate that is None.")
    if not drop_qubit:
        _, m, p = _qrepr(qubit).measure(_idx(qubit)[0], observable, modify=True)
        return m, p
    qrepr, m, p = _qrepr(qubit).measure_discard(_idx(qubit)[0], observable)
    qubit.qstate.drop_qubit(qubit, new_qrepr=qrepr)
    if not discard:
        QState([qubit], get_qstate_formalism().create_in_basis([m], observable))
    return m, p


def gmeasure(qubits, meas_operators, check_operators=False):
    """Make a general qubit measurement with the specified measurement operators.

    Parameters
    ----------
    qubits : :obj:`~netsquid.qubits.qubit.Qubit` or list of :obj:`~netsquid.qubits.qubit.Qubit`
        Single qubit or a list of qubits upon which to operate. The order of the
        qubits with respect to the measurement operators is important.
    meas_operators : list (or tuple) of :class:`~netsquid.qubits.operators.Operator` or :class:`~netsquid.qubits.operators.Operator`
        List of measurement operators with which to measure, or a single observable, i.e. an operator with valid projectors.
        The measurement operatators ``M_i`` should satisfy the
        completeness relation ``Sum_i^N M_i.H * M_i = I``.
        The single observable should be a Hermitian matrix.
    check_operators : bool, optional
        If a list of operators is specified, then check if they satisfy the
        completeness relation defined above. Default if False.

    Returns
    -------
    int
        Index of the measurement operator that succeeded.
    float
        Probability with which this result occurred.

    Raises
    ------
    TypeError
        If ``qubits`` is not a qubit or a non-empty list of qubits.
    ValueError
        If the ``meas_operators`` are checked and don't satisfy the constraints.

    Notes
    -----
        Unlike the ``measure`` function, a general measurement will not
        split the qubit from its shared quantum state.

    Example
    -------

    >>> q1, q2 = create_qubits(2)  # |00> state
    >>> # Measure q2 in the standard basis (P_0 = |0><0|, P_1 = |1><1|)
    >>> P_0, P_1 = ops.Z.projectors
    >>> gmeasure(q2, [P_0, P_1])
    (0, 1.0)

    Perform a parity check in the standard basis on both qubits.

    >>> gmeasure([q1, q2], ops.Z ^ ops.Z)
    (0, 1.0)

    """
    if check_operators:
        if isinstance(meas_operators, ops.Operator):
            if not meas_operators.is_hermitian:
                raise ValueError("The observable is not Hermitian.")
        else:
            ops.check_measurement_operators(meas_operators)
    qubits = _to_qubits_list(qubits, combine=True)
    _, m, p = _qrepr(qubits).gmeasure(_idx(qubits), meas_operators, modify=True)
    return m, p


def discard(qubit):
    """Drop a qubit from its shared quantum state.

    The discarded qubit's quantum state will become ``None``.

    Parameters
    ----------
    qubit : :obj:`~netsquid.qubits.qubit.Qubit`
        Qubit to discard.

    Notes
    -----
        How a qubit is *dropped* is formalism specific.
        In the DM formalism the qubit is traced out.
        In the pure state formalisms the qubit is first measured (in
        standard basis) if the quantum state is shared with one or more
        qubits, which is equivalent to calling ``measure(q, discard=True)``.

        If ``qubit`` is the only qubit in the quantum state, then
        the only action is to assign its quantum state to ``None``.
        For efficiency the quantum state object itself is not modified
        in this case but simply assumed to be lost i.e. it will continue
        to reference the dropped qubit (but not vice versa).

    Examples
    --------
    Discard the second qubit from the ``|00>`` state in the:

    >>> q1, q2 = create_qubits(2)  # separate states |0>, |0>
    >>> combine_qubits([q1, q2])  # state |00>
    >>> discard(q2)
    >>> print(q1.qstate)
    QState([Qubit('QS#14-0')])
    >>> print(q2.qstate)
    None

    """
    qstate = qubit.qstate
    if qstate is None:
        # Qubit is already discarded
        return
    if qstate.num_qubits > 1:
        qstate.drop_qubit(qubit)
    else:
        qubit.qstate = None


def operate(qubits, operator):
    """Apply an operator to a qubit or list of qubits.

    Parameters
    ----------
    qubits : :obj:`~netsquid.qubits.qubit.Qubit` or list of :obj:`~netsquid.qubits.qubit.Qubit`
        Single qubit or a list of qubits upon which to operate. The order of the
        qubits with respect to the operator is important.
    operator : :obj:`~netsquid.qubits.operators.Operator`
        Operator acting on qubits

    Raises
    ------
    TypeError
        If ``qubits`` is not a qubit or a non-empty list of qubits.

    Notes
    -----
        Before an operator is applied to more than one qubit, the quantum states
        of the interacting qubits are combined.

    Examples
    --------
    Let us begin by operating on single qubits. In the first case we
    use the predefined ``X`` operator and in the second case we define a new
    operator ``R``. Note that the quantum states of the two qubits remain separate.

    >>> q1, q2 = create_qubits(2)  # |00>
    >>> operate(q1, ops.X)
    >>> print(reduced_dm(q1))
    [[0.+0.j 0.+0.j]
     [0.+0.j 1.+0.j]]
    >>> theta = np.pi/8
    >>> R = ops.Operator("R", np.cos(theta) * np.eye(2) - 1j * np.sin(theta) * ops.X.arr)
    >>> operate(q2, R)
    >>> print(reduced_dm(q2))
    [[0.85355339+0.j         0.        +0.35355339j]
     [0.        -0.35355339j 0.14644661+0.j        ]]

    Note that we could have also created the same rotation operator ``R`` with
    ``Operator`` arithmetic using already defined operators:

    >>> R = np.cos(theta) * ops.I - 1j * np.sin(theta) * ops.X
    >>> print(R.name)
    (((0.92)*I)-((0.00+0.38j)*X))

    or, simpler yet, using the helper function

    >>> R = ops.create_rotation_op(theta, (0, 1, 0))
    >>> print(R.name)
    R_y[0.39]

    Next we operate on two qubits simultaneously. The order that we specify the
    qubits should match with the desired vector space action of the operator.
    For instance, for the ``CNOT`` gate the first qubit is the *control* and the
    second the *target*. After the qubit operation below, the two qubits will
    share the same quantum state.

    >>> q1, q2 = create_qubits(2)  # |00>
    >>> operate(q1, ops.H)
    >>> operate([q1, q2], ops.CNOT)
    >>> print(reduced_dm([q1, q2]))
    [[0.5+0.j 0. +0.j 0. +0.j 0.5+0.j]
     [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
     [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
     [0.5+0.j 0. +0.j 0. +0.j 0.5+0.j]]
    >>> print(q1.qstate == q2.qstate)
    True
    >>> # Check fidelity with the Bell state b00 = (|00> + |11>)/sqrt(2)
    >>> round(fidelity([q1, q2], ketstates.b00), 2)
    1.0

    To illustrate that the order in which the qubits are listed matters, let us
    reverse the order of both the Hadamard and the CNOT gates. Thus ``q2`` is
    now the *control* and ``q1`` the target. We observe that this gives the same
    Bell state. Finally we apply a controlled Z gate (``CZ``).

    >>> q1, q2 = create_qubits(2)
    >>> operate(q2, ops.H)
    >>> operate([q2, q1], ops.CNOT)
    >>> # Check fidelity with the Bell state b00 = (|00> + |11>)/sqrt(2)
    >>> round(fidelity([q1, q2], ketstates.b00), 2)
    1.0
    >>> operate([q1, q2], ops.CZ)
    >>> print(reduced_dm([q1, q2]))
    [[ 0.5+0.j  0. +0.j  0. +0.j -0.5+0.j]
     [ 0. +0.j  0. +0.j  0. +0.j  0. +0.j]
     [ 0. +0.j  0. +0.j  0. +0.j  0. +0.j]
     [-0.5+0.j  0. +0.j  0. +0.j  0.5+0.j]]
    >>> # Check fidelity with the Bell state b10 = (|00> - |11>)/sqrt(2)
    >>> round(fidelity([q1, q2], ketstates.b10), 2)
    1.0

    We can construct controlled gates using the controlled variable.
    For example, a controlled T gate:

    >>> CT = ops.T.ctrl
    >>> print(CT.name)
    CT
    >>> print(CT.description)
    Controlled T gate: q1 = control, q2 = target
    >>> print(CT.arr)  # doctest: +SKIP
    [[1.        +0.j         0.        +0.j         0.        +0.j       0.       +0.j        ]
     [0.        +0.j         1.        +0.j         0.        +0.j       0.       +0.j        ]
     [0.        +0.j         0.        +0.j         1.        +0.j       0.       +0.j        ]
     [0.        +0.j         0.        +0.j         0.        +0.j       0.7071...+0.7071...j]]

    """
    qubits = _to_qubits_list(qubits, combine=True)
    try:
        _qrepr(qubits).operate(_idx(qubits), operator, modify=True)
    except MissingQRepr:  # DEPRECATED
        qubits[0].qstate.operate_qubits(qubits, operator)


def stochastic_operate(qubits, operators, p_weights=None):
    r"""Stochastically apply a list of quantum operators to a qubit or list of qubits.

    Parameters
    ----------
    qubits : :obj:`~netsquid.qubits.qubit.Qubit` or list of :obj:`~netsquid.qubits.qubit.Qubit`
        Single qubit or a list of qubits upon which to operate. The order of the
        qubits with respect to the operator is important.
    operators : list of :obj:`~netsquid.qubits.operators.Operator`
        Operators acting on the qubit(s).
    p_weights : tuple of float or None, optional
        Probability weights for each operator. Must sum to 1.
        The default (None) is a uniform distribution.

    Raises
    ------
    ValueError
        If given ``p_weights`` tuple has an incorrect length or does not
        sum to one.

    Notes
    -----
        For the given list of ``operators``, \\(O_i\\), and ``p_weights``,
        \\(p_i\\) (with \\(\\sum_i p_i = 1\\)),
        this function does the quantum operation
        \\(\\varepsilon(\\rho) = \\sum_i p_i O_i \\rho \\, O_i^\\dagger\\).
        For pure states it mimics this operation by randomly selecting a
        single operator \\(O_i\\) with probability \\(p_i\\) and applying
        this to the ``qubits``.

    Examples
    --------

    Depolarize a qubit using the density matrix formalism:

    >>> set_qstate_formalism(QFormalism.DM)
    >>> q1, = create_qubits(1)
    >>> stochastic_operate(q1, [ops.I, ops.X, ops.Y, ops.Z])
    >>> print(reduced_dm(q1))
    [[0.5+0.j 0. +0.j]
     [0. +0.j 0.5+0.j]]

    Randomly select a controlled gate:

    >>> q1, q2 = create_qubits(2)
    >>> operate([q1, q2], ops.X ^ ops.X)
    >>> stochastic_operate([q1, q2], [ops.CNOT, ops.CZ], p_weights=(1/3, 2/3))

    """
    rng = simtools.get_random_state()
    if p_weights is None:
        p_weights = tuple(np.ones(len(operators)) / len(operators))
    elif len(p_weights) != len(operators) or \
            not cymath.isclose(sum(p_weights), 1., rel_tol=1e-5):
        raise ValueError("Invalid 'p_weights' parameter specified")
    qubits = _to_qubits_list(qubits, combine=True)
    if _qrepr(qubits).supports_mixed_states:
        multi_operate(qubits, operators, p_weights)
    else:
        operator = operators[rng.choice(len(operators), p=p_weights)]
        operate(qubits, operator)


def multi_operate(qubits, operators, weights=None):
    r"""Apply a (weighted) list of quantum operators to a qubit or list of qubits.

    This represents a general quantum operation on an ensemble of states.
    The result will in general be a mixed state. In the case of a pure state formalism,
    a pure state is sampled from the resulting mixed state.

    Parameters
    ----------
    qubits : :obj:`~netsquid.qubits.qubit.Qubit` or list of :obj:`~netsquid.qubits.qubit.Qubit`
        Single qubit or a list of qubits upon which to operate. The order of the
        qubits with respect to the operator is important.
    operators : list or tuple of :obj:`~netsquid.qubits.operators.Operator`
        Operators acting on qubits.
    weights : tuple of float or None, optional
        Weights for each operator. The default (``None``) is for each
        operator to have a weight of 1.

    Raises
    ------
    ValueError
        If given ``weights`` tuple has an incorrect length.
    NotImplementedError
        If quantum state formalism can not represent ensemble states e.g.
        :obj:`~netsquid.qubits.qubitapi.QFormalism.STAB`.

    Notes
    -----
        For the given list of ``operators``, \\(O_i\\), and abitrary ``weights``,
        \\(w_i\\), this function peforms the quantum operation
        \\(\\varepsilon(\\rho) = \\sum_i w_i O_i \\rho \\, O_i^\\dagger\\).

    Examples
    --------
    Do generalized amplitude damping:

    >>> set_qstate_formalism(QFormalism.DM)
    >>> q1, = create_qubits(1)
    >>> gamma = 0.1
    >>> prob = 0.5
    >>> a = np.sqrt(gamma)
    >>> b = np.sqrt(1-gamma)
    >>> E0 = ops.Operator("E0_AD", [[1, 0], [0, b]])
    >>> E1 = ops.Operator("E1_AD", [[0, a], [0, 0]])
    >>> E2 = ops.Operator("E2_AD", [[b, 0], [0, 1]])
    >>> E3 = ops.Operator("E3_AD", [[0, 0], [a, 0]])
    >>> multi_operate(q1, [E0, E1, E2, E3],
    ...               weights=(prob, prob, 1 - prob, 1 - prob))
    >>> print(reduced_dm(q1))
    [[0.95+0.j 0.  +0.j]
     [0.  +0.j 0.05+0.j]]

    """
    qubits = _to_qubits_list(qubits, combine=True)
    qrepr = _qrepr(qubits)
    if not qrepr.supports_universal and get_qstate_formalism().supports_universal:
        # This can occur if all qubits were prepared in STAB
        convert_qstate(qubits)
    qrepr.multi_operate(_idx(qubits), list(operators), weights, modify=True)


def fidelity(qubits, reference_state, squared=False):
    r"""Calculate fidelity between the qubit(s) quantum state and a reference state.

    The fidelity is calculated as implemented in each QState formalism (KET, DM, STAB)
    These formalisms implement a method to reduce a large qubit state to a size matching the reference state.

    Parameters
    ----------
    qubits : :obj:`~netsquid.qubits.qubit.Qubit` or list of :obj:`~netsquid.qubits.qubit.Qubit`
        Single qubit or a list of qubits for which to compute the fidelity.
        If a list of qubits is given their quantum states will be *combined*
        (in the given order) to ensure they share the same state.
    reference_state : :class:`numpy.array` or :class:`netsquid.qubits.qrepr.QRepr`
        Ket vector or density matrix state with which fidelity is computed.
    squared : bool, optional
        If True the squared Fidelity is returned.

    Returns
    -------
    float
        Fidelity (or squared fidelity).

    Raises
    ------
    ValueError
        If number of qubits does not match size of reference state.

    Notes
    -----
        This function does not modify the qubits or the shared quantum state.


    Examples
    --------
    We check the fidelity of two qubits with the Bell state
    ``b00 = (|00> + |11>)/sqrt(2)`` before and after applying
    the necessary gate operations.

    >>> q1, q2 = create_qubits(2)  # |00>
    >>> round(fidelity([q1, q2], ketstates.b00), 2)
    0.71
    >>> operate(q1, ops.H)
    >>> operate([q1, q2], ops.CNOT)
    >>> round(fidelity([q1, q2], ketstates.b00), 2)
    1.0

    We can also check the fidelity for the reduced state of each
    qubit separately.

    >>> round(fidelity(q1, ketstates.s0, squared=True), 2)
    0.5
    >>> round(fidelity(q2, ketstates.s0, squared=True), 2)
    0.5

    """
    # Note: Some dmstates aren't complex, which is noticed in the cython type checking.
    # Therefore we have to use astype(complex) here.
    if not isinstance(reference_state, QRepr):
        # warn_deprecated("Providing numpy arrays to the fidelity is deprecated. Use a QRepr instance instead.",
        #                 "qubitapi.fidelity.no_qrepr")
        if isinstance(reference_state, np.ndarray) and reference_state.dtype != complex:
            reference_state = reference_state.astype(complex, casting='safe')
        if reference_state.ndim == 1:
            reference_state = reference_state.reshape((reference_state.shape[0], 1))
        if reference_state.shape[0] == reference_state.shape[1]:
            reference_state = DenseDMRepr(reference_state)
        else:
            reference_state = KetRepr(reference_state)
    qubits = _to_qubits_list(qubits, combine=True)
    return _qrepr(qubits).fidelity(reference_state, _idx(qubits), squared=squared)


def exp_value(qubits, operator):
    """Calculate the expectation value of an operator over the (reduced) state of qubits.

    Parameters
    ----------
    qubits : :obj:`~netsquid.qubits.qubit.Qubit` or list of :obj:`~netsquid.qubits.qubit.Qubit`
        Single qubit or a list of qubits with which to compute the expectation value.
        The order of the qubits with respect to the operator is important.
    operator : :obj:`~netsquid.qubits.operators.Operator`
        Operator to calculate expectation value for.

    Returns
    -------
    float
        Expectation value for the given operator.

    Notes
    -----
        This function does not modify the qubits or the shared quantum state.

    """
    qubits = _to_qubits_list(qubits, combine=True)
    if len(qubits) != operator.num_qubits:
        raise ValueError("Number of qubits must match the size of the operator.")
    dm = reduced_dm(qubits)
    return np.real(np.trace(dm @ operator.arr))


def apply_pauli_noise(qubit, p_weights):
    """Randomly apply pauli noise to a qubit according probability weights.

    Parameters
    ----------
    qubit : :obj:`~netsquid.qubits.qubit.Qubit`
        Qubit to apply noise to.
    p_weights : tuple of float
        Probability distribution of I, X, Y, Z given that qubit depolarizes.
        Must sum to one.

    Raises
    ------
    ValueError
        If the specified `p_weights` typle has incorrect length or does
        not sum to one.

    Notes
    -----
        In formalisms that do not support ensemble states the depolarization is
        simulated by randomly applying Pauli gate operations.

    Example
    -------
    Apply ``X`` noise to a qubit with a 50% probability:

    >>> q1, = create_qubits(1)
    >>> apply_pauli_noise(q1, (0.5, 0.5, 0, 0))

    Depolarize a qubit:

    >>> q1, = create_qubits(1)
    >>> apply_pauli_noise(q1, (1/4, 1/4, 1/4, 1/4))

    Depolarize a qubit with probability ``p``:

    >>> q1, = create_qubits(1)
    >>> p = 0.8
    >>> apply_pauli_noise(q1, (1. - 0.75 * p, 0.25 * p, 0.25 * p, 0.25 * p))

    """
    if len(p_weights) != 4:
        raise ValueError(f"length of p_weights tuple {len(p_weights)} != 4")
    if not math.isclose(sum(p_weights), 1.):
        raise ValueError("Input probability weights (p_weights) must sum to one")
    if not np.all([0 <= p <= 1. for p in p_weights]):
        raise ValueError("Input probability weights cannot be negative")
    if p_weights[0] == 1.0:
        return
    if get_qstate_formalism().supports_mixed_states:
        stochastic_operate([qubit], [ops.I, ops.X, ops.Y, ops.Z], p_weights=p_weights)
    else:
        if simtools.get_random_state().random_sample() > p_weights[0]:
            if p_weights[0] == 0.:
                p_weights = p_weights[1:]
            else:
                p_weights = np.array(p_weights[1:])
                p_weights = tuple(p_weights / np.sum(p_weights))
            stochastic_operate([qubit], [ops.X, ops.Y, ops.Z], p_weights=p_weights)


def depolarize(qubit, prob):
    """Randomly depolarize a qubit with a given probability.

    Parameters
    ----------
    qubit : :obj:`~netsquid.qubits.qubit.Qubit`
        Qubit to depolarize.
    prob : float
        Probability that qubit will depolarize.

    Notes
    -----
        Note that the probability that the quantum state of a qubit is actually
        changed due to application of a Pauli operator is 0.75 * `prob`.

        In formalisms that do not support ensemble states the depolarization is
        simulated by randomly applying Pauli gate operations.

    Example
    -------
    >>> q1, = create_qubits(1)  # |0>
    >>> depolarize(q1, prob=0.80)

    """
    if math.isclose(prob, 0.):
        return
    if prob < 0 or prob > 1:
        raise ValueError(f"{prob} is an invalid probability.")
    p = 0.25 * prob
    apply_pauli_noise(qubit, (1. - 3 * p, p, p, p))


def dephase(qubit, prob):
    """Randomly apply a Z-gate to a qubit with a given probability.

    Parameters
    ----------
    qubit : :obj:`~netsquid.qubits.qubit.Qubit`
        Qubit to dephase.
    prob : float
        Probability :math:`p` that Z gate is applied.

    Notes
    -----
        :math:`p = 0` is no dephasing, :math:`p = 1` is a Z gate,
        :math:`p = 1/2` is max dephasing.

        In formalisms that do not support ensemble states the dephasing is
        simulated by randomly applying a Z gate operation.

    Example
    -------
    >>> q1, = create_qubits(1)  # |0>
    >>> dephase(q1, prob=0.80)

    """
    if math.isclose(prob, 0.):
        return
    if prob < 0 or prob > 1:
        raise ValueError(f"{prob} is an invalid probability.")
    apply_pauli_noise(qubit, (1. - prob, 0, 0, prob))


def amplitude_dampen(qubit, gamma, prob=1., cache=True, cache_precision=-1):
    r"""Apply generalized amplitude damping to a qubit.

    Parameters
    ----------
    qubit : :obj:`~netsquid.qubits.qubit.Qubit`
        Qubit to apply amplitude damping to.
    gamma : float
        Damping parameter.
    prob : float, optional
        Probability :math:`p` defining the stationary state (see below).
    cache : bool, optional
        Whether to use caching to speed up computation for identical
        ``gamma`` and ``prob`` parameters (up to ``cache_precision``).
    cache_precision : int, optional
        Decimal place precision to round caching lookup parameter(s) to.
        If set to -1, then no rounding is performed.

    Raises
    ------
    NotImplementedError
        If the quantum state formalism being used cannot do amplitude damping e.g.
        stabilizer state formalisms.

    Notes
    -----
        Applies the quantum operation :math:`\varepsilon(\rho) = \sum_i E_i \rho \, E_i^\dagger`
        using the elements:

        .. math::
            \begin{aligned}
            E_0 &= \sqrt{p}\begin{bmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{bmatrix} &
            E_1 &= \sqrt{p}\begin{bmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{bmatrix} \\
            E_2 &= \sqrt{1-p}\begin{bmatrix} \sqrt{1 - \gamma} & 0 \\ 0 & 1 \end{bmatrix} &
            E_3 &= \sqrt{1-p}\begin{bmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{bmatrix}
            \end{aligned}

        The stationary state that is unaffected by
        generalized amplitude damping is

        .. math::
            \rho_\infty = \begin{bmatrix} p & 0 \\ 0 & 1-p \end{bmatrix},

        such that :math:`\varepsilon(\rho_\infty) = \rho_\infty`.

        For pure quantum state formalisms (i.e. :obj:`~netsquid.qubits.qubitapi.QFormalism.KET`)
        an ancilla bit is used to apply the amplitude damping.

    Example
    -------
    >>> set_qstate_formalism(QFormalism.DM)
    >>> q1, = create_qubits(1)  # |0>
    >>> amplitude_dampen(q1, gamma=0.1, prob=1)
    >>> print(reduced_dm(q1))
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]
    >>> operate([q1], ops.X)  # -> |1>
    >>> amplitude_dampen(q1, gamma=0.1, prob=1)
    >>> print(reduced_dm(q1))
    [[0.1+0.j 0. +0.j]
     [0. +0.j 0.9+0.j]]

    """
    krausops, weights = calc_amplitude_dampen_ops(gamma, prob, cache, cache_precision)
    multi_operate([qubit], krausops, weights=tuple(weights))


def apply_dda_noise(qubits, depol=0., deph=0., ampl=0.):
    """Applies depolarising, dephasing and amplitude-damping noise to a qubit.

    Parameters
    ----------
    qubit : :obj:`~netsquid.qubits.qubit.Qubit` or list of :obj:`~netsquid.qubits.qubit.Qubit`
        Qubit(s) to apply (independent) noise to.
    depol : float
        Depolarising parameter.
    deph : float
        Dephasing parameter.
    ampl : float
        Damping parameter.

    Raises
    ------
    NotImplementedError
        If amplitude damping parameter non-zero and the quantum state formalism being used
        cannot do amplitude damping e.g. stabilizer state formalisms.

    Notes
    -----
        Does not include generalized amplitude-damping noise. The order of applying the
        noise is fixed: depolarize, dephase then amplitude dampen, which might matter.

    """
    qubits = _to_qubits_list(qubits)
    for qubit in qubits:
        depolarize(qubit, prob=depol)
        dephase(qubit, prob=deph)
        if ampl > 0:
            # Let amplitude_dampen raise exception if stabilizer state is used
            amplitude_dampen(qubit, gamma=ampl, prob=1)


def delay_depolarize(qubit, depolar_rate, delay):
    r"""Randomly depolarize a qubit for a delay with a given depolarization rate.

    Parameters
    ----------
    qubit : :obj:`~netsquid.qubits.qubit.Qubit`
        Qubit to depolarize.
    depolar_rate : float
        Depolarization rate :math:`R` for qubit (see notes below) [Hz].
        The probability of not depolarizing follows an exponential distribution, such
        that a ``delay`` of 1/``depolar_rate`` gives a depolarizing probability of 63%.
    delay : float
        Time delay :math:`\Delta t` during which qubit could have depolarized [ns]
        (see notes below).

    Returns
    -------
    float
        Probability that the qubit was depolarized.

    Notes
    -----
        The depolarizing probability is calculated as :math:`p = 1 - \exp(-\Delta t\, R)`,
        where :math:`R` is the depolarization rate and :math:`\Delta t` the delay.

        In formalisms that do not support ensemble states the depolarization is
        simulated by randomly applying Pauli gate operations.

    Example
    -------
    >>> q1, = create_qubits(1)  # |0>
    >>> delay_depolarize(q1, depolar_rate=0.10, delay=10e9)  # delay of 10 seconds
    0.6321...

    """
    if depolar_rate < 0:
        raise ValueError(f"depolar rate {depolar_rate} should be non-negative.")
    # prob = 1. - poisson.cdf(0, delay * depolar_rate * 1e-9)  # equivalent
    prob = 1. - math.exp(- delay * depolar_rate * 1e-9)
    depolarize(qubit, prob)
    return prob


def delay_dephase(qubit, dephase_rate, delay):
    r"""Randomly dephase a qubit for a delay with a given dephasing rate.

    Parameters
    ----------
    qubit : :obj:`~netsquid.qubits.qubit.Qubit`
        Qubit to dephase.
    dephase_rate : float
        Dephasing rate :math:`R` for qubit (see notes below) [Hz].
        The probability of not dephasing follows an exponential distribution, such
        that a ``delay`` of 1/``dephase_rate`` gives a dephasing probability of 63%.
    delay : float
        Time delay :math:`\Delta t` during which qubit could have dephased [ns]
        (see notes below).

    Returns
    -------
    float
        Probability that the qubit was dephased.

    Notes
    -----
        The dephasing probability is calculated as :math:`p = 1 - \exp(-\Delta t\, R`,
        where :math:`R` is the dephasing rate and :math:`\Delta t` the delay.

        In formalisms that do not support ensemble states the dephasing is
        simulated by randomly applying Pauli gate operations.

    Example
    -------
    >>> q1, = create_qubits(1)  # |0>
    >>> delay_dephase(q1, dephase_rate=0.10, delay=10e9)  # delay of 10 seconds
    0.6321...

    """
    if dephase_rate < 0:
        raise ValueError(f"dephase rate {dephase_rate} should be non-negative.")
    prob = 1. - math.exp(- delay * dephase_rate * 1e-9)
    dephase(qubit, prob)
    return prob


def reduced_dm(qubits):
    """Returns the reduced density matrix for the given qubit(s).

    This is the result of tracing out all other qubits from
    the shared quantum state of ``qubits``.

    This is the recommended way to get the density matrix for a collection of qubits.

    Parameters
    ----------
    qubits : :obj:`~netsquid.qubits.qubit.Qubit` or list of :obj:`~netsquid.qubits.qubit.Qubit`
        List of qubits to find the reduced state for. The order of the qubits
        will define the vector space for the reduced state.

    Returns
    -------
    :obj:`numpy.array`
        Density matrix representing the reduced density matrix.

    Notes
    -----
        The combined qubits are first *combined* if necessary.
        It does not otherwise modify the qubits or the shared quantum state.

    Example
    -------
    >>> q1, q2 = create_qubits(2)
    >>> operate(q1, ops.H)
    >>> operate([q1, q2], ops.CNOT)
    >>> print(reduced_dm(q1))
    [[0.5+0.j 0. +0.j]
     [0. +0.j 0.5+0.j]]

    """
    qubits = _to_qubits_list(qubits, combine=True)
    return _qrepr(qubits).reduced_dm(_idx(qubits))
