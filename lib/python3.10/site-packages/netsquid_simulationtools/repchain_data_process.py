import time
import pandas
import itertools
import numpy as np
from argparse import ArgumentParser
from copy import deepcopy

from netsquid_simulationtools.repchain_data_combine import combine_data
from netsquid_simulationtools.repchain_data_plot import plot_qkd_data, plot_fidelity_rate, plot_teleportation
from netsquid_simulationtools.repchain_dataframe_holder import RepchainDataFrameHolder
from netsquid_simulationtools.repchain_data_functions import end_to_end_fidelity, estimate_duration_per_success
from netsquid_simulationtools.process_qkd import qber, estimate_bb84_secret_key_rate_from_data
from netsquid_simulationtools.process_teleportation import teleportation_fidelity_optimized_over_local_unitaries_from_data, \
    estimate_average_teleportation_fidelity_from_data, determine_teleportation_fidelities_of_xyz_eigenstates_from_data, \
    estimate_minimum_teleportation_fidelity_over_xyz_eigenstates_from_data


def process_data(raw_data_dir="raw_data", suffix=".pickle", output="processed_data.pickle",
                 csv_output_filename="output.csv", process_duration=True, process_bb84=False,
                 process_teleportation=False, process_fidelity=False, plot_processed_data=True):
    """Read different RepchainDataFrameHolder files with a common suffix and process them.

    Which data processing functions are used can be specified using the different `process_` arguments of this function.

    Parameters
    ----------
    raw_data_dir : str (optional)
        Directory containing the data (pickle files containing
        :class:`~netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`s) to be processed.
    suffix : str (optional)
        Common ending of the files to be combined.
        Files with this suffix are first combined into a single `RepchainDataFrameHolder`, then stored,
        and then this combined `RepchainDataFrameHolder` is processed.
    output : str (optional)
        Filename with which to save the combined `RepchainDataFrameHolder`.
    csv_output_filename : str (optional)
        Filename with which to save the processed data.
        The processed data is a `DataFrame` saved as a CSV file.
    process_duration : bool (optional)
        Set true if entanglement-generation duration should be calculated.
    process_bb84 : bool (optional)
        Set true if Quantum Bit Error Rate (QBER) should be estimated from the data, and used to calculate the
        achievable secret-key rate when performing the BB84 protocol.
        Requires `basis_A`, `basis_B`, `outcome_A` and `outcome_B` to be columns of the dataframe associated
        with the `RepchainDataFrameHolder`.
    process_teleportation : bool (optional)
        Set true if quality of quantum teleportation using the entanglement in the data should be calculated.
        The calculated quantities include average teleportation fidelity, minimum teleportation fidelity over
        Pauli X, Y and Z eigenstates, and average teleportation fidelity if first an optimization over local unitary
        transformations is performed.
        Required `state` to be one of the columns of the dataframe associated with the `RepchainDataFrameHolder`.
    process_fidelity : bool (optional)
        Set true if end-to-end fidelity should be calculated.
        Requires `state` to be a column of the dataframe associated with the `RepchainDataFrameHolder`.
    plot_processed_data : Boolean (optional)
        If True, plots the processed data with the plotting scripts available to the used processing types (default).
        If False, nothing is plotted.

    """
    processing_functions = []
    if process_duration:
        processing_functions.append(process_data_duration)
    if process_bb84:
        processing_functions.append(process_data_bb84)
    if process_teleportation:
        processing_functions.append(process_data_teleportation_fidelity)
    if process_fidelity:
        processing_functions.append(process_data_fidelity)

    start_time = time.time()

    combined_data = combine_data(raw_data_dir=raw_data_dir, suffix=suffix, output=output, save_output_to_file=True)

    processed_data = process_repchain_dataframe_holder(combined_data, processing_functions=processing_functions)

    # sort data by first scan_param
    processed_data.sort_values(by=combined_data.varied_parameters[0], inplace=True)

    # save processed data
    processed_data.to_csv(csv_output_filename, index=False)

    runtime = time.time() - start_time

    print(f"processed files in {raw_data_dir} with suffix {suffix} in {runtime} seconds\n"
          f"Used processing functions: {processing_functions}")

    if plot_processed_data:
        filename = csv_output_filename
        scan_param_name = combined_data.varied_parameters[0]
        if process_bb84 and process_duration:
            plot_qkd_data(filename=filename, scan_param_name=scan_param_name,
                          scan_param_label=scan_param_name)
        if process_teleportation:
            plot_teleportation(filename=filename, scan_param_name=scan_param_name,
                               scan_param_label=scan_param_name, show_duration=process_duration)
        if process_fidelity:
            plot_fidelity_rate(filename=filename, scan_param_name=scan_param_name,
                               scan_param_label=scan_param_name)


def process_repchain_dataframe_holder(repchain_dataframe_holder, processing_functions, **kwargs):
    """Process a single :class:`RepchainDataFrameHolder` using some processing functions.

    The `repchain_dataframe_holder`'s dataframe is split into multiple dataframes, one for each unique value of the
    varied parameters.
    Each dataframe is passed to each function in `processing_functions`, together with a dictionary `sim_params`
    which contains all the simulation parameters used to obtain the data in that specific dataframe
    (i.e. all the baseline parameters of `repchain_dataframe_holder` and the values of all varied parameters
    corresponding to that part of the data).
    The processing function is then expected to return a dictionary with processed information;
    the keys of the dictionary should be the names of the different calculated quantities,
    while the values should be the calculated values.

    The only processed quantity automatically calculated by this function is the total number of successes
    corresponding to each unique value of the varied parameters,
    which is included under the name number_of_successes.

    Parameters
    ----------
    repchain_dataframe_holder : :class:`~netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`
        Raw data to be processed.
    processing_functions : list of functions
        Functions used to process dataframes with data corresponding to a specific value of the simulation parameters.
        Each function should take `dataframe` (a `DataFrame` with data as specified in the documentation of
        :class:`~RepchainDataFrameHolder`) and `sim_params` (a dictionary with simulation parameter values
        used to obtain data in `dataframe`) as arguments,
        and output a dictionary holding names of processed quantities as keys and their calculated values as values.

    Returns
    -------
    :class:`pd.DataFrame`
        Dataframe containing processed data.
        The dataframe has one row per unique value of varied parameters.
        The columns are `number_of_successes`, `generation_duration_unit`, any varied parameters
        and the processed quantities returned by the different `processing_functions`.

    Notes
    -----
    The number_of_successes includes successes for which the basis choices are unequal.

    """

    # access data in container
    base_params = repchain_dataframe_holder.baseline_parameters
    varied_params = repchain_dataframe_holder.varied_parameters

    # TODO: think of a better way to deal with theses varied parameters
    for key in ["generation_duration", "number_of_successes", "sample_file", "current_round", "time_stamp",
                "node_distance"]:
        if key in varied_params:
            varied_params.remove(key)

    varied_params_unique_values = []
    for varied_param in varied_params:
        varied_params_unique_values.append(repchain_dataframe_holder.dataframe[varied_param].unique().tolist())

    processed_data = pandas.DataFrame()
    for values in itertools.product(*varied_params_unique_values):

        sim_params = deepcopy(base_params)
        df = repchain_dataframe_holder.dataframe
        varied_params_with_values = {}
        for (varied_param, value) in zip(varied_params, values):
            sim_params.update({varied_param: value})
            varied_params_with_values.update({varied_param: value})
            df = df[df[varied_param] == value]
        df.drop(columns=varied_params)

        if df.empty:
            continue

        processed_data_for_these_parameters = {"number_of_successes": [len(df.index)]}
        for processing_function in processing_functions:
            processed_data_for_these_parameters_by_this_processing_function = processing_function(dataframe=df,
                                                                                                  sim_params=sim_params,
                                                                                                  **kwargs)
            processed_data_for_these_parameters.update(processed_data_for_these_parameters_by_this_processing_function)

        processed_data_for_these_parameters.update(varied_params_with_values)
        processed_data_for_these_parameters.update(
            {"generation_duration_unit": repchain_dataframe_holder.baseline_parameters["generation_duration_unit"]})
        processed_data = processed_data.append(pandas.DataFrame(processed_data_for_these_parameters),
                                               ignore_index=True)

    return processed_data


def process_data_duration(dataframe, sim_params, **kwargs):
    """Calculate the entanglement-generation duration using entanglement-generation data.

    The processed quantities calculated by this function are:
    * duration_per_success, the average amount of time required to distribute a single entangled state.
    * duration_per_success_error, the standard error in the estimate of `duration_per_success`.

    This function is designed to be used with
    :func:`~netsquid_simulationtools.repchain_data_process.process_repchain_dataframe_holder()`

    Parameters
    ----------
    dataframe : :class:`pandas.DataFrame`
        Entanglement-generation data in the format specified in the documentation of
        :class:`~netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`,
        but all data must have been generated using the same simulation parameters.
    sim_params : dict
        Simulation parameters and their values used to generate the data held by `dataframe`.
        Keys are names of simulation parameters, values are parameter values.

    Returns
    -------
    dict
        Processed quantity names as keys, their calculated values as values.

    Notes
    -----
    For more information about how the generation duration per success is estimated, see
    :class:`netsquid_simulationtools.repchain_data_functions.estimate_duration_per_success`.


    """
    duration_per_success, duration_per_success_error = estimate_duration_per_success(dataframe=dataframe)
    processed_data = {
        "duration_per_success": [duration_per_success],
        "duration_per_success_error": [duration_per_success_error]}
    return processed_data


def process_data_bb84(dataframe, sim_params, sifting_factor=1, **kwargs):
    """Calculate achievable secret-key rate using BB84 and Quantum-Bit Error Rate (QBER) from entanglement data.

    The processed quantities calculated by this function are:
    * sk_rate, the estimated secret-key rate achievable with the BB84 protocol using the entanglement-generation
    statistics in the data.
    * sk_rate_upper_bound, upper error bound on estimated secret-key rate.
    * sk_rate_lower_bound, lower error bound on estimated secret-key rate.
    * sk_error, standard error in estimated secret-key rate.
    * Qber_z, the Quantum Bit Error Rate (QBER) in the Z basis.
    * Qber_z_error, the standard error in the Quantum Bit Error Rate (QBER) in the Z basis.
    * Qber_x, the Quantum Bit Error Rate (QBER) in the X basis.
    * Qber_x_error, the standard error in the Quantum Bit Error Rate (QBER) in the X basis.

    If `length` is defined in `sim_params`, the following quantities are also calculated:
    * plob_bound, the PLOB bound on achievable secret-key rate without quantum repeater.
    * tgw_bound, the TGW bound on achievable secret-key rate without quantum repeater.

    This function is designed to be used with
    :func:`~netsquid_simulationtools.repchain_data_process.process_repchain_dataframe_holder()`

    Parameters
    ----------
    dataframe : :class:`pandas.DataFrame`
        Entanglement-generation data in the format specified in the documentation of
        :class:`~netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`,
        but all data must have been generated using the same simulation parameters.
    sim_params : dict
        Simulation parameters and their values used to generate the data held by `dataframe`.
        Keys are names of simulation parameters, values are parameter values.
    sifting_factor : float (optional)
        Probability that both Alice and Bob use the same measurement basis.
        Defaults to 1, representing fully-asymmetric BB84 (see notes below for explanation).
        If both bases are chosen individually at random, `sifting_factor` should be set to 0.5.

    Returns
    -------
    dict
        Processed quantity names as keys, their calculated values as values.

    Notes
    -----
    For more information about how the secret-key rate is estimated, see
    :class:`netsquid_simulationtools.repchain_data_functions.estimate_bb84_secret_key_rate_from_data`.

    For more information about how the QBER is estimated, see
    :class:`netsquid_simulationtools.repchain_data_functions.qber`.

    For more information about the PLOB and TGW bounds, see
    :func:`~netsquid_simulationtools.repchain_data_process.secret_key_capacity()`.

    """

    # calculate qber in X and Z
    qber_z, qber_z_error = qber(dataframe, "Z")
    qber_x, qber_x_error = qber(dataframe, "X")

    # determine secret key rate in [bits/attempt]
    secret_key_rate, skr_min, skr_max, skr_error = \
        estimate_bb84_secret_key_rate_from_data(dataframe, sifting_factor=sifting_factor)

    # put all results in dictionary
    processed_data = {
        "sk_rate": [secret_key_rate],
        "sk_rate_upper_bound": [skr_max],
        "sk_rate_lower_bound": [skr_min],
        "sk_error": [skr_error],
        "Qber_z": [qber_z],
        "Qber_z_error": [qber_z_error],
        "Qber_x": [qber_x],
        "Qber_x_error": [qber_x_error]}

    if "length" in sim_params:
        length = sim_params["length"]
        attenuation = sim_params["attenuation"] if "attenuation" in sim_params else 0.25
        if "attenuation" not in sim_params:
            print("NOTE: capacity is based on default attenuation of 0.25 db/km, which may not correspond to "
                  "attenuation used in simulations.")

        # calculate secret-key capacity
        plob_bound, tgw_bound = secret_key_capacity(length=length, attenuation=attenuation)

        processed_data.update({"plob_bound": [plob_bound],
                               "tgw_bound": [tgw_bound]})

    return processed_data


def process_data_teleportation_fidelity(dataframe, sim_params, **kwargs):
    """Calculate the achievable average fidelity of quantum teleportation achievable with data.

    The processed quantities calculated by this function are:
    * teleportation_fidelity_average, the teleportation fidelity averaged over all possible information (input) states,
    using "standard protocol" described in documentation of
    :func:`~netsquid-simulationtools.repchain_data_functions.determine_teleportation_output_state()`.
    * teleportation_fidelity_average_error, standard error in teleportation_fidelity_average.
    * teleportation_fidelity_minimum_xyz_eigenstate_index, eigenstate of Pauli X, Y and Z operators for which
    the teleportation fidelity is the smallest if the state is used as information state.
    Is a value of :class:`netsquid_simulationtools.repchain_data_functions.XYZEigenstateIndex`.
    * teleportation_fidelity_minimum_xyz_eigenstates, the teleportation fidelity minimized over all eigenstates of
    the Pauli X, Y and Z operators as information state. This is the teleportation fidelity when using
    teleportation_fidelity_minimum_xyz_eigenstate_index as information state.
    (which is generally not the same as the minimum over all information states).
    * teleportation_fidelity_minimum_xyz_eigenstates_error, the standard error in
    teleportation_fidelity_minimum_xyz_eigenstates.
    * teleportation_fidelity_average_optimized_local_unitaries, the teleportation fidelity averaged over all possible
    information (input) states using a protocol that is optimized over local unitary transformations.
    For more information, see the documentation of
    :func:`~netsquid_simulationtools.repchain_data_functions.teleportation_fidelity_optimized_over_local_unitaries()`.

    This function is designed to be used with
    :func:`~netsquid_simulationtools.repchain_data_process.process_repchain_dataframe_holder()`

    Parameters
    ----------
    dataframe : :class:`pandas.DataFrame`
        Entanglement-generation data in the format specified in the documentation of
        :class:`~netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`,
        but all data must have been generated using the same simulation parameters.
    sim_params : dict
        Simulation parameters and their values used to generate the data held by `dataframe`.
        Keys are names of simulation parameters, values are parameter values.

    Returns
    -------
    dict
        Processed quantity names as keys, their calculated values as values.

    """
    fidelities_xyz_dataframe = determine_teleportation_fidelities_of_xyz_eigenstates_from_data(dataframe)
    teleportation_fidelity_average, teleportation_fidelity_average_error = \
        estimate_average_teleportation_fidelity_from_data(fidelities_xyz_dataframe)
    teleportation_fidelity_minimum_xyz_eigenstate_index, teleportation_fidelity_minimum_xyz_eigenstates, \
        teleportation_fidelity_minimum_xyz_eigenstates_error = \
        estimate_minimum_teleportation_fidelity_over_xyz_eigenstates_from_data(fidelities_xyz_dataframe)
    teleportation_fidelity_average_optimized_local_unitaries = \
        teleportation_fidelity_optimized_over_local_unitaries_from_data(dataframe)

    processed_data = {
        "teleportation_fidelity_average": teleportation_fidelity_average,
        "teleportation_fidelity_average_error": teleportation_fidelity_average_error,
        "teleportation_fidelity_minimum_xyz_eigenstate_index": teleportation_fidelity_minimum_xyz_eigenstate_index,
        "teleportation_fidelity_minimum_xyz_eigenstates": teleportation_fidelity_minimum_xyz_eigenstates,
        "teleportation_fidelity_minimum_xyz_eigenstates_error": teleportation_fidelity_minimum_xyz_eigenstates_error,
        "teleportation_fidelity_average_optimized_local_unitaries":
            teleportation_fidelity_average_optimized_local_unitaries
    }
    return processed_data


def process_data_fidelity(dataframe, sim_params, **kwargs):
    """Calculate average end-to-end fidelity of entangled states.

    The processed quantities calculated by this function are:
    * fidelity, the average fidelity of end-to-end states to expected Bell state.
    * fidelity_error, the standard error in the estimate of the fidelity.

    This function is designed to be used with
    :func:`~netsquid_simulationtools.repchain_data_process.process_repchain_dataframe_holder()`

    Parameters
    ----------
    dataframe : :class:`pandas.DataFrame`
        Entanglement-generation data in the format specified in the documentation of
        :class:`~netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`,
        but all data must have been generated using the same simulation parameters.
    sim_params : dict
        Simulation parameters and their values used to generate the data held by `dataframe`.
        Keys are names of simulation parameters, values are parameter values.

    Returns
    -------
    dict
        Processed quantity names as keys, their calculated values as values.

    """

    # calculate fidelity and its error
    fidelity, fidelity_error = end_to_end_fidelity(dataframe)

    # put all results in dictionary
    processed_data = {
        "fidelity": [fidelity],
        "fidelity_error": [fidelity_error],
    }

    return processed_data


def secret_key_capacity(length, attenuation=0.25):
    """Calculate Pirandola-Laurenza-Ottaviani-Banchi (PLOB) bound for lossy channels using two-way communication
    (DOI: 10.1038/ncomms15043) (i.e. the maximum attainable for direct transmission) and the Takeoka-Guha-
    Wilde (TGW) bound.

    Parameters
    ----------
    length : float
        Distance between end nodes performing QKD.
    attenuation : float
        Fiber loss in dB/km.

    Return
    ------
    plob_bound : float
        PLOB bound for secret-key-rate capacity for direct transmission in [bits/channel_use].
    tgw_bound : float
        Takeoka-Guha-Wilde bound for direct transmission in [bits/channel_use].

    """
    if length < 0:
        raise ValueError("Length cannot be negative.")
    transmissivity = 0.9999 if np.isclose(length, 0) else np.power(10, - attenuation * length / 10)
    plob_bound = - np.log2(1 - transmissivity)  # sk capacity , PLOB bound in [bits/channel_use]
    tgw_bound = np.log2((1 + transmissivity) / (1 - transmissivity))  # tgw bound [bits/channel_use]

    return plob_bound, tgw_bound


def convert_rdfh_with_number_of_rounds(rdfh):
    """Convert RepchainDataFramaHolder with outdated column name (`number_of_rounds` to `generation_duration`).
    As of version 1.0.0, the way time scales are handled for RepchainDataFrameHolder has been changed. To unify
    time scales across different platforms, RDFHs now have a column called `generation_duration`. The unit of this
    column must be specified in the baseline parameters with `generation_duration_unit` and would usually have values
    such as "seconds" or "rounds". To be able to work with old RDFHs containing `number_of_rounds` column, this method
    can be used to fix how time scales are handled.

    Parameters
    ----------
    rdfh : :obj:`nlblueprint.repchain_dataframe_holder.RepchainDataFrameHolder`
        RepchainDataFrameHolder containing column with `number_of_rounds`.

    Return
    ------
    rdfh : :obj:`nlblueprint.repchain_dataframe_holder.RepchainDataFrameHolder`
        RepchainDataFrameHolder with abovementioned column converted into `generation_duration`.
    """
    if not isinstance(rdfh, RepchainDataFrameHolder):
        raise ValueError("Input parameter is not RepchainDataFrameHolder")
    if 'number_of_rounds' not in rdfh.dataframe.columns:
        raise ValueError("Number of rounds is not in the RepchainDataFrameHolder")

    # rename column with `number_of_rounds`
    rdfh.dataframe = rdfh.dataframe.rename(columns={'number_of_rounds': 'generation_duration'})

    # update baseline parameters
    rdfh.baseline_parameters.update({'generation_duration_unit': 'rounds'})

    return rdfh


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-d", "--raw_data_dir", required=False, type=str, default="raw_data",
                        help="Directory containing the raw data (i.e. pickled RepchainDataFrameHolders)"
                             "that should be processed.")
    parser.add_argument("-s", "--suffix", required=False, type=str, default=".pickle",
                        help="Common suffix of all the raw-data files in raw_data_dir that should be processed.")
    parser.add_argument("-o", "--output", required=False, type=str, default="combined_data.pickle",
                        help="File to store combined data in. This is a pickled RepchainDataFrameHolder including"
                             "all the results of all the RepchainDataFrameHolders included in processing.")
    parser.add_argument("-f", "--filename_csv", required=False, type=str, default="output.csv",
                        help="File to store processing results in using CSV format.")
    parser.add_argument("--bb84", dest="process_bb84", action="store_true", help="Perform BB84 processing.")
    parser.add_argument("-t", "--teleportation", dest="process_teleportation_fidelity", action="store_true",
                        help="Process achievable quantum-teleportation fidelity.")
    parser.add_argument("--fidelity", dest="process_fidelity", action="store_true", help="Perform end-to-end "
                                                                                         "fidelity processing.")
    parser.add_argument("--plot", dest="plot", action="store_true", help="Plot the processing results.")

    args = parser.parse_args()
    process_data(raw_data_dir=args.raw_data_dir, suffix=args.suffix, output=args.output,
                 csv_output_filename=args.filename_csv, process_bb84=args.process_bb84,
                 process_teleportation=args.process_teleportation_fidelity,
                 process_fidelity=args.process_fidelity, plot_processed_data=args.plot)
