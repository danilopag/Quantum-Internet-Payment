from statistics import mean, stdev
import numpy as np
import pandas as pd
from netsquid import BellIndex
from netsquid.qubits.ketstates import b01, b11, b00, b10

from netsquid_simulationtools.linear_algebra import xyz_eigenstates, _convert_to_density_matrix, pauli_operators, \
    pauli_x, pauli_y, pauli_z, identity, _perform_pauli_correction, _fidelity_between_single_qubit_states, \
    XYZEigenstateIndex
from netsquid_simulationtools.repchain_data_functions import _expected_target_state, estimate_density_matrix_from_data


def estimate_average_teleportation_fidelity_from_data(fidelities_dataframe):
    """Estimate the fidelity of quantum teleportation, averaged over all possible information states, based on data.

    For a description of the quantum-teleportation protocol considered here, see the documentation of
    :func:`~netsquid-simulationtools.repchain_data_functions.determine_teleportation_output_state()`.

    This function estimates the average teleportation fidelity achievable using entanglement distribution
    as described by the data.
    Here, the average is taken over all possible information states
    (where the information state is the state sent by the quantum-teleportation protocol).
    To calculate the average over all possible information states, the average over Pauli X, Y and Z eigenstates
    is used, which gives the same result (assuming the average is taken over the uniform Haar measure).
    This is shown in the following paper:
    Fidelity of single qubit maps
    Bowdrey et al, 2002, https://www.sciencedirect.com/science/article/pii/S0375960102000695

    This function requires the teleportation fidelities for the different Pauli-operator eigenstates to be
    precalculated.
    It has been designed to be used together with the function
    :func:`netsquid_simulationtools.repchain_data_functions.
    determine_teleportation_fidelities_of_xyz_eigenstates_from_data`;
    the output of that function can be used as input for this function.

    The Pauli X, Y and Z eigenstates are enumerated in the documentation of
    :func:`~netsquid_simulationtools.repchain_data_functions.
    estimate_minimum_teleportation_fidelity_over_xyz_eigenstates_from_data()`

    Parameters
    ----------
    fidelities_dataframe : pandas.DataFrame
        Dataframe containing teleportation fidelities of X, Y and Z eigenstates.
        The dataframe should have one column per X, Y, Z eigenstate
        (using :class:`netsquid_simulationtools.repchain_data_functions.XYZEigenstateIndex` as column titles).
        Each row in the dataframe is considered a single simulation result.
        The dataframe should have exactly the format as outputted by
        :func:`netsquid_simulationtools.repchain_data_functions.
        determine_teleportation_fidelities_of_xyz_eigenstates_from_data`,
        and they are in fact intended to be used together.

    Returns
    -------
    float
        Average teleportation fidelity estimated from entanglement-distribution data.
    float
        Standard error of average teleportation fidelity.
        This is the standard deviation of the mean for the average teleportation fidelity.

    """
    # Average over all eigenstates per result in the data
    average_fidelity_per_result = fidelities_dataframe.mean(axis=1).tolist()
    average_fidelity = mean(average_fidelity_per_result)
    average_fidelity_error = stdev(average_fidelity_per_result) / np.sqrt(len(average_fidelity_per_result))

    return average_fidelity, average_fidelity_error


def estimate_minimum_teleportation_fidelity_over_xyz_eigenstates_from_data(fidelities_dataframe):
    """Estimate minimum teleportation fidelity over the set of eigenstates of Pauli X, Y and Z from data.

    For a description of the quantum-teleportation protocol considered here, see the documentation of
    :func:`~netsquid-simulationtools.repchain_data_functions.determine_teleportation_output_state()`.

    For each of all the eigenstates of the Pauli X, Y and Z operators as information state,
    the achievable teleportation fidelity is estimated using entanglement-distribution data.
    The minimum of all the results is returned by this function.
    Note that this is in general not the same as the teleportation fidelity minimized over all possible information
    states.

    This function requires the teleportation fidelities for the different eigenstates to be precalculated.
    It has been designed to be used together with the function
    :func:`netsquid_simulationtools.repchain_data_functions.
    determine_teleportation_fidelities_of_xyz_eigenstates_from_data`;
    the output of that function can be used as input for this function.

    The Pauli X, Y and Z eigenstates are:
    * Pauli X operator, eigenvalue +1: |+> = (|0> + |1>) / sqrt(2).
    * Pauli X operator, eigenvalue -1: |-> = (|0> - |1>) / sqrt(2).
    * Pauli Y operator, eigenvalue +1: (|0> + i * |1>) / sqrt(2).
    * Pauli Y operator, eigenvalue -1: (|0> - i * |1>) / sqrt(2).
    * Pauli Z operator, eigenvalue +1: |0>.
    * Pauli Z operator, eigenvalue -1: |1>.

    Parameters
    ----------
    fidelities_dataframe : pandas.DataFrame
        Dataframe containing teleportation fidelities of X, Y and Z eigenstates.
        The dataframe should have one column per X, Y, Z eigenstate
        (using :class:`netsquid_simulationtools.repchain_data_functions.XYZEigenstateIndex` as column titles).
        Each row in the dataframe is considered a single simulation result.
        The dataframe should have exactly the format as outputted by
        :func:`netsquid_simulationtools.repchain_data_functions.
        determine_teleportation_fidelities_of_xyz_eigenstates_from_data`,
        and they are in fact intended to be used together.

    Returns
    -------
    :class:`netsquid_simulationtools.repchain_data_functions.XYZEigenstateIndex`
        XYZEigenstateIndex of the Pauli-operator eigenstate for which the teleportation fidelity is minimal.
    float
        Minimum teleportation fidelity over eigenstates of Pauli X, Y and Z operators.
    float
        Standard error of minimum teleportation fidelity over eigenstates of Pauli X, Y and Z operators.
        This is the standard deviation of the mean for the teleportation fidelity of the state for which the
        teleportation fidelity is smallest.

    """
    average_fidelities = dict(fidelities_dataframe.mean(axis=0))

    state_with_smallest_fidelity = XYZEigenstateIndex(min(average_fidelities, key=average_fidelities.get))
    smallest_fidelity = average_fidelities[state_with_smallest_fidelity]
    smallest_fidelity_error = (stdev(fidelities_dataframe[state_with_smallest_fidelity]) /
                               np.sqrt(len(fidelities_dataframe[state_with_smallest_fidelity])))
    return state_with_smallest_fidelity, smallest_fidelity, smallest_fidelity_error


def determine_teleportation_fidelities_of_xyz_eigenstates_from_data(dataframe):
    """Determine teleportation fidelity per simulation result of Pauli X, Y and Z eigenstates.

    Calculates the teleportation fidelities for each of the X, Y and Z eigenstates for each of the entangled states
    provided by the data.
    This can subsequently be used to calculate quantities such as the average teleportation fidelity
    (see :func:`netsquid_simulationtools.repchain_data_functions.estimate_average_teleportation_fidelity_from_data`),
    or the minimum teleportation fidelity over the X, Y and Z eigenstates
    (see :func:`netsquid_simulationtools.
    repchain_Data_functions.estimate_minimum_teleportation_fidelity_over_xyz _eigenstates_from_data`).

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing entanglement-distribution data.
        Should contain the column `state` containing two-qubit quantum states
        (either length-4 vector or 4x4 density matrix).
        For specification of how the data should be structured, see the documentation of
        :class:`~netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`.

    Returns
    -------
    pandas.DataFrame
        Dataframe with one column per X, Y, Z eigenstate
        (using :class:`netsquid_simulationtools.repchain_data_functions.XYZEigenstateIndex` as column titles)
        and one row per result contained by `dataframe`.

    """

    information_states = xyz_eigenstates

    if "state" not in dataframe.columns:
        raise ValueError("Data does not contain state information.")

    fidelities = {information_state_label: [] for information_state_label in information_states}
    for index, row in dataframe.iterrows():
        index_of_expected_bell_state = _expected_target_state(row)
        for information_state_label, information_state in information_states.items():
            fidelity = determine_teleportation_fidelity(information_state=information_state,
                                                        resource_state=row["state"],
                                                        bell_index=index_of_expected_bell_state)
            fidelities[information_state_label].append(fidelity)

    return pd.DataFrame(fidelities)


def estimate_teleportation_fidelity_from_data(dataframe, information_state):
    """Estimate the teleportation fidelity of a specific information state using entanglement-generation data.

    First determines the effective end-to-end (mixed) state between the end nodes by combining all state information
    in the data.
    Then, it calculates the teleportation fidelity with that state as resource state for a specific information state.
    For a description of the quantum-teleportation protocol considered here, see the documentation of
    :func:`~netsquid-simulationtools.repchain_data_functions.determine_teleportation_output_state()`.

    Note that there is a statistical error in the average fidelity calculated by this function,
    which is the result of using finite statistics to determine the end-to-end density matrix.
    The uncertainty in the estimate of the density matrix propagates into this function.
    The uncertainty is not quantified here.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing entanglement-distribution data.
        For specification of how the data should be structured, see the documentation of
    information_state : numpy.array
        Single-qubit state to be sent using quantum teleportation.
        Can either be a length-2 vector of 2x2 density matrix.

    Returns
    -------
    float
        Teleportation fidelity for `information_state`.

    Notes
    -----
    If there is need for this, the function can be extended to also return standard error.
    To this end, instead of first calculating the effective state and calculating the teleportation fidelity
    using this state as resource state, the teleportation fidelity should be determined per result in the data.
    This gives statistics from which the standard error can be calculated.

    """
    resource_state = estimate_density_matrix_from_data(dataframe)
    return determine_teleportation_fidelity(information_state=information_state,
                                            resource_state=resource_state,
                                            bell_index=BellIndex.B00)


def teleportation_fidelity_optimized_over_local_unitaries_from_data(dataframe):
    """Determine optimal teleportation fidelity optimized over local unitary operations from entanglement data.

    First determines the effective end-to-end (mixed) state between the end nodes by combining all state information
    in the data.
    Then, it calculates the optimal teleportation fidelity possible with that state when optimizing over local
    unitary transformations at the end nodes.
    For more information about this protocol, see the documentation of
    :func:`netsquid_simulationtools.repchain_data_functions.teleportation_fidelity_optimized_over_local_unitaries`.

    Note that there is a statistical error in the average fidelity calculated by this function,
    which is the result of using finite statistics to determine the end-to-end density matrix.
    The uncertainty in the estimate of the density matrix propagates into this function.
    The uncertainty is not quantified here.


    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing entanglement-distribution data.
        For specification of how the data should be structured, see the documentation of
        :class:`~netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`.

    Returns
    -------
    float
        Maximum achievable teleportation fidelity ("squared" definition) when allowing for local unitary tranformations,
        which is the fidelity of the output state to the information state,
        averaged over all possible information states.

    """
    state = estimate_density_matrix_from_data(dataframe)
    return teleportation_fidelity_optimized_over_local_unitaries(resource_state=state)


def teleportation_fidelity_optimized_over_local_unitaries(resource_state):
    """Determine optimal teleportation fidelity optimized over local unitary operations.

    Before "standard" quantum teleportation is performed (for description, see documentation of
    :func:`~netsquid-simulationtools.repchain_data_functions.determine_teleportation_output_state()`),
    local unitaries are applied to the resource state such that the fidelity,
    averaged over all possible information states, is maximized.
    This maximized fidelity is a function only of the fidelity of the resource state to the "closest"
    maximally-entangled state, and the local unitaries only rotate the state such that the closest
    maximally-entangled state becomes the reference state assumed by the teleportation procedure
    (here, this is |Phi+> == (|00> + |11>) / sqrt(2)).

    The linear relation between the maximized teleportation fidelity and maximal fidelity to a maximally-entangled state
    was first described in the paper
    General teleportation channel, singlet fraction, and quasidistillation.
    Horodecki, Horodecki and Horodecki, 1999, https://link.aps.org/doi/10.1103/PhysRevA.60.1888.

    For this function, we use the convenient calculational method presented in Proposition 1 of
    Fidelity deviation in quantum teleportation with a two-qubit state
    Ghosal et al, 2020, https://doi.org/10.1088/1751-8121/ab6ede
    This calculation is based on the T matrix of the resource state.
    The paper also describes how the standard deviation in the fidelity when using this teleportation strategy
    can be calculated (also using the T matrix).

    It is also possible to optimize the teleportation fidelity further, but this involves using non-unitary operations.
    By using filtering, fidelity can be boosted using a trace-preserving map (by preparing a known product
    state each time when filtering "fails".
    A dynamical programming recipe for finding the optimal trace-preserving map is presented here:
    Optimal Teleportation with a Mixed State of Two Qubits
    Verstraete and Verschelde, 2003, https://link.aps.org/doi/10.1103/PhysRevLett.90.097901.
    The fidelity determined by this function only optimizes over local unitary operations and thus does not consider
    strategies such as filtering.


    Parameters
    ----------
    resource_state : numpy.array
        Two-qubit (entangled) state used as resource state for quantum teleportation.

    Returns
    -------
    float
        Maximum achievable teleportation fidelity ("squared" definition) when allowing for local unitary tranformations,
        which is the fidelity of the output state to the information state,
        averaged over all possible information states.

    """
    resource_state = _convert_to_density_matrix(resource_state)
    t_matrix = np.zeros([3, 3])
    for i, j in np.ndindex(3, 3):
        t_matrix[i][j] = np.trace(resource_state @ np.kron(pauli_operators[i], pauli_operators[j]))

    eigenvalues_t, eigenvectors_t = np.linalg.eig(t_matrix)
    eigenvalues_t = [abs(eigenvalue) for eigenvalue in eigenvalues_t]
    eigenvalues_sum = sum(eigenvalues_t)
    det_t = np.linalg.det(t_matrix)

    if det_t <= 0:
        return (1 + eigenvalues_sum / 3) / 2
    else:
        return (1 + max([eigenvalues_sum - 2 * eigenvalue for eigenvalue in eigenvalues_t]) / 3) / 2


def determine_teleportation_output_state(information_state, resource_state):
    """Determine output state of quantum teleportation with specific information state and resource state.

    In the quantum teleportation, an information state is transmitted using an entangled bipartite resource state.
    Here, we consider the following exact protocol.
    First, a Bell-state measurement is performed between the information state and one qubit in the resource state.
    Pauli corrections depending on the measurement outcome (see below for the exact relation)
    are performed on the remaining qubit in the resource state.
    If the resource state is maximally entangled state |phi+> = (|00> + |11>) / sqrt(2),
    the information state is now held by the qubit that underwent the corrections.

    It is assumed that no conditioning on the outcome of the Bell-state measurement is performed.
    Therefore, the teleportation output state as returned by this function is a mixture of the states
    resulting from the different Bell-state measurement outcomes.

    If the maximally entangled state that has the largest fidelity to the resource state is not the
    state |phi+> = (|00> + |11>) / sqrt(2), the teleportation fidelity can be increased by performing local
    unitary operations that make |phi+> the closest entangled state.
    For more information about maximizing teleportation fidelity, see the documentation of
    :func:`~netsquid_simulationtools.repchain_data_functions.teleportation_fidelity_optimized_over_local_unitaries()`.

    Pauli corrections as function of Bell-state measurement outcome:
    * outcome |phi+> = (|00> + |11>) / sqrt(2): no corrections.
    * outcome |psi+> = (|01> + |10>) / sqrt(2) = X |psi+>: Pauli X correction.
    * outcome |psi-> = (|01> - |10>) / sqrt(2) = Y |psi+>: Pauli Y correction.
    * outcome |phi-> = (|00> - |11>) / sqrt(2) = Z |psi+>: Pauli Z correction.

    Parameters
    ----------
    information_state : numpy.array
        Single-qubit state to be sent using quantum teleportation.
        Can either be a length-2 vector or 2x2 density matrix.
    resource_state : numpy.array
        Two-qubit (entangled) state used as resource state for quantum teleportation.
        Can either be a length-4 vector or a 4x4 density matrix.

    Returns
    -------
    numpy.array
        Output state of quantum teleportation as 2x2 density matrix.

    """

    # first convert all states to density matrices
    information_state = _convert_to_density_matrix(information_state)
    resource_state = _convert_to_density_matrix(resource_state)

    # initial density matrix is tensor product of information state and resource state
    initial_dm = np.kron(information_state, resource_state)

    output_state = 0
    for bell_state, correction in zip([b00, b01, b11, b10], [None, pauli_x, pauli_y, pauli_z]):
        effective_measurement_operator = np.kron(bell_state, identity)
        unnormalized_state_without_correction = \
            effective_measurement_operator.conj().T @ initial_dm @ effective_measurement_operator
        if np.isclose(np.trace(unnormalized_state_without_correction), 0):
            continue
        if correction is not None:
            unnormalized_state = correction @ unnormalized_state_without_correction @ correction.conj().T
        else:
            unnormalized_state = unnormalized_state_without_correction
        output_state += unnormalized_state

    return output_state


def determine_teleportation_fidelity(information_state, resource_state, bell_index=BellIndex.B00):
    """Calculate fidelity of quantum teleportation with specific information state and resource state.

    For a description of the quantum-teleportation protocol considered here, see the documentation of
    :func:`~netsquid-simulationtools.repchain_data_functions.determine_teleportation_output_state()`.
    This function determines the fidelity of the teleported state to the input state.

    The teleportation protocol works best if the closest maximally entangled state to the resource state is
    |Phi+> = (|00> + |11>) / sqrt(2).
    If the resource state is expected to be closer to another Bell state,
    a local Pauli correction should be performed before performing the teleportation protocol.
    This correction has the goal to bring the resource state closer to |Phi+> as possible.
    This function takes that correction into account.
    Note that since the correction is performed on the second qubit in the resource state, which is interpreted
    as the "remote" qubit, performing the correction on the entangled state is equivalent to performing the correction
    on the output qubit after finishing teleportation.

    The fidelity is averaged over all possible Bell-state-measurement outcomes that can be obtained while performing
    quantum teleportation.

    Parameters
    ----------
    information_state : numpy.array
        Single-qubit state to be sent using quantum teleportation.
        Can either be a length-2 vector or 2x2 density matrix.
    resource_state : numpy.array
        Two-qubit (entangled) state used as resource state for quantum teleportation.
        Can either be a length-4 vector or a 4x4 density matrix.
    bell_index : :class:`netsquid.qubits.ketstates.BellState`
        Index of Bell state that the resource state is expected to have the largest fidelity to.

    Returns
    -------
    float
        Fidelity of output state to information state.

    """
    corrected_resource_state = _perform_pauli_correction(state=resource_state,
                                                         bell_index=bell_index)
    output_state = determine_teleportation_output_state(information_state=information_state,
                                                        resource_state=corrected_resource_state)
    return _fidelity_between_single_qubit_states(output_state, information_state)
