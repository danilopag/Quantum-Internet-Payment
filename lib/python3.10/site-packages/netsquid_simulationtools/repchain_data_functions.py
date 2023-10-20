"""Unified processing functions for repeater chain simulations

Note: For now only use with Bell states as source states.
"""
from statistics import mean, stdev
import numpy as np
from netsquid import BellIndex
from scipy import sparse
import netsquid.qubits.qubitapi as qapi
from netsquid.qubits.ketstates import b01, b11, b00, b10

from netsquid_physlayer.detectors import BSMOutcome

from netsquid_simulationtools.linear_algebra import _perform_pauli_correction


def end_to_end_fidelity(dataframe):
    """Function that computes the average end-to-end fidelity from the input dataframe.

    Note: The Dataframe has to contain the end-to-end states.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing simulation data (especially end-to-end states and midpoint outcomes).

    Returns
    -------
    avg_fidelity : float
        Average fidelity measured for the states in the DataFrame.
    fidelity_error : floar
        Error of the average fidelity for the DataFrame.

    """
    fidelities = []
    for index, row in dataframe.iterrows():
        if row["state"] is None:
            raise ValueError("Can't calculate fidelity without end-to-end state.")
        else:
            dm = row["state"]

        if sparse.issparse(dm):
            # convert sparse to dense
            dm = dm.toarray()

        # create qubits with state
        num_qubits = int(np.log2(dm.shape[0]))
        qubits = qapi.create_qubits(num_qubits=num_qubits)
        qapi.assign_qstate(qubits, dm)

        # get index from expected target state
        bell_index = _expected_target_state(row)
        # convert index to reference ket
        bell_index_to_ket = [b00, b01, b11, b10]
        reference_ket = bell_index_to_ket[bell_index]

        if dm.shape[0] == 256:
            # create relevant target states
            if bell_index == 1:
                # target state: (|01,00;00,01> + |00,01;01,00>)/sqrt(2)
                reference_ket = np.array([[0. + 0.j]]*256)
                reference_ket[65] = 1/np.sqrt(2.) + 0.j
                reference_ket[20] = 1/np.sqrt(2.) + 0.j
            elif bell_index == 2:
                # target state: (|01,00;00,01> - |00,01;01,00>)/sqrt(2)
                reference_ket = np.array([[0. + 0.j]]*256)
                reference_ket[65] = 1/np.sqrt(2.) + 0.j
                reference_ket[20] = - 1/np.sqrt(2.) + 0.j
            else:
                # discarding these events
                break
        elif dm.shape[0] == 16:
            # map kets to multi photon encoding
            zero = np.array([[0. + 0.j]]*16)
            zero[0] = reference_ket[0]
            zero[1] = reference_ket[1]
            zero[4] = reference_ket[2]
            zero[5] = reference_ket[3]
            reference_ket = zero

        # calculate fidelity
        f = qapi.fidelity(qubits, reference_ket, squared=True)

        # append
        fidelities.append(f)
    # calculate mean and std
    avg_fidelity = mean(fidelities)
    fidelity_error = stdev(fidelities) / np.sqrt(len(fidelities))

    return avg_fidelity, fidelity_error


def estimate_density_matrix_from_data(dataframe):
    """Construct state mixture delivered by repeater chain as density matrix based on data.

    The states delivered on end nodes by a repeater chain typically differ per run and are statistical in nature.
    These statistics can be described using ensemble-based quantum formalisms, such as the density-matrix formalism.
    This function considers all the delivered states described in the data and expresses the mixture as a
    density matrix.

    Before "adding" a state to the ensemble, first a Pauli correction is applied in accordance with the midpoint
    outcomes of the repeater chain.
    Thus, in the absence of noise, this function will always return the Bell state |Phi+> = (|00> + |11>) / sqrt(2).

    Note that there is statistical uncertainty in estimating the density matrix, as the amount of data used to
    characterize the mixture of states is only finite.
    The uncertainty is however not quantified by this function.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing entanglement-distribution data.
        For specification of how the data should be structured, see the documentation of
        :class:`~netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`.

    Returns
    -------
    numpy.array
        Density matrix describing the end-to-end (mixed) state created by the repeater chain.

    """
    if "state" not in dataframe.columns:
        raise ValueError("Data does not contain state information.")
    state = 0
    for index, row in dataframe.iterrows():
        state_this_result = row["state"]
        if state_this_result.shape == (4, 1):
            state_this_result = state_this_result @ state_this_result.conj().T
        expected_bell_index = _expected_target_state(row)
        state_this_result = _perform_pauli_correction(state=state_this_result,
                                                      bell_index=expected_bell_index)
        state += state_this_result
    state /= len(dataframe.index)
    return state


def estimate_duration_per_success(dataframe):
    """Estimate the number of attempts per distributed Bell state and its error.

    This does not include the effect of sifting.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing entanglement-distribution data.
        For specification of how the data should be structured, see the documentation of
        :class:`~netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`.

    Returns
    -------
    attempts_per_success : float
        Estimate in number of attempts required to distribute a single Bell state.
    attempts_per_success_error : float
        Standard error in the estimate of the number of attempts requried to distribute a single Bells state.

    """

    successful_bits_generated = len(dataframe.index)
    try:
        generation_durations = dataframe["generation_duration"].tolist()
    except KeyError:
        generation_durations = dataframe["number_of_attempts"].tolist()
    total_duration = sum(generation_durations)
    duration_per_success = total_duration / successful_bits_generated
    duration_per_success_error = stdev(generation_durations) / np.sqrt(total_duration)

    return duration_per_success, duration_per_success_error


def _expected_target_state(row):
    """ Function that computes the expected target state shared between Alice and Bob for a single row of the input
    DataFrame.

    This is done by looking at all swap and midpoint outcomes and applying the corresponding Pauli corrections to the
    |Phi_+> state.

    Note:
    As all currently all sources create the same Bell states and there is always an even number of sources in a
    repeater chain (2 per elementary link), this holds for any Bell state the sources create..

    Parameters
    ----------
    row : pandas.Series
        Row of the DataFrame containing simulation data.

    Returns
    -------
    bell_index : :class:`netsquid.qubits.ketstates.BellIndex`
        Bell index of the expected target state that Alice and Bob share.

    """
    # extract all midpoint and swap outcomes and combine them in a list
    all_outcomes = []

    for column_name, value in row.iteritems():
        split_column_name = column_name.split("_")
        if len(split_column_name) > 1:
            if column_name.split("_")[1] == "outcome" and column_name.split("_")[0] in ["midpoint", "swap"]:
                if isinstance(value, BSMOutcome):
                    value = value.bell_index
                if not isinstance(value, float) and not isinstance(value, int):
                    raise ValueError(f"Outcomes in Dataframe are of type {type(value)} but "
                                     f"should be float.")
                all_outcomes.append(value)

    # count number of Pauli corrections and using: sigma_i . sigma_i = identity
    x = all_outcomes.count(BellIndex.PSI_PLUS) % 2
    y = all_outcomes.count(BellIndex.PSI_MINUS) % 2
    z = all_outcomes.count(BellIndex.PHI_MINUS) % 2

    # create 3d array that maps x,y,z to bell index
    # x | y | z | effect
    # ------------------
    # 0   0   0 |  0 > 0
    # 1   0   0 |  0 > 1
    # 0   1   0 |  0 > 2
    # 1   1   0 |  0 > 3
    # 0   0   1 |  0 > 3
    # 1   0   1 |  0 > 2
    # 0   1   1 |  0 > 1
    # 1   1   1 |  0 > 0

    bell_index = [[[BellIndex.PHI_PLUS, BellIndex.PHI_MINUS], [BellIndex.PSI_MINUS, BellIndex.PSI_PLUS]],
                  [[BellIndex.PSI_PLUS, BellIndex.PSI_MINUS], [BellIndex.PHI_MINUS, BellIndex.PHI_PLUS]]]

    # when using atomic ensembles with presence-absence encoding, two chains have to be used. The expected correlation
    # depends on the target state of both chains
    if 'chain_2_midpoint_0' in row.keys():
        all_outcomes_chain_2 = []
        for column_name, value in row.iteritems():
            split_column_name = column_name.split("_")
            if len(split_column_name) > 1:
                if column_name.split("_")[0] == "chain" and column_name.split("_")[1] == "2" and \
                   column_name.split("_")[2] in ["midpoint", "swap"]:
                    if isinstance(value, BSMOutcome):
                        value = value.bell_index
                    if not isinstance(value, float) and not isinstance(value, int):
                        raise ValueError(f"Outcomes in Dataframe are of type {type(value)} but "
                                         f"should be float.")
                    all_outcomes_chain_2.append(value)
        # check of both chains had the same length
        if len(all_outcomes_chain_2) != len(all_outcomes):
            raise ValueError(f"Chain 1 had {len(all_outcomes)} while chain 2 had {len(all_outcomes_chain_2)}")
        x_2 = all_outcomes_chain_2.count(BellIndex.PSI_PLUS) % 2
        y_2 = all_outcomes_chain_2.count(BellIndex.PSI_MINUS) % 2
        z_2 = all_outcomes_chain_2.count(BellIndex.PHI_MINUS) % 2
        return bell_index[x][y][z], bell_index[x_2][y_2][z_2]

    return bell_index[x][y][z]
