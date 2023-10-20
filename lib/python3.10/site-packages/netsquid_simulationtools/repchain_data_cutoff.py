import warnings

from netsquid_simulationtools.repchain_dataframe_holder import RepchainDataFrameHolder
from netsquid_simulationtools.repchain_data_process import process_repchain_dataframe_holder, process_data_bb84
from netsquid_simulationtools.repchain_data_combine import combine_data
from copy import deepcopy
import pickle
import time
import os
import matplotlib.pyplot as plt


def implement_cutoff_duration(repchain_dataframe_holder, max_duration, use_default_method=True):
    """Function which changes a repchain_dataframe_holder such that it is as if it were performed using a cutoff time.

    Parameters
    ----------
    repchain_dataframe_holder : :object:`~.repchain_dataframe_holder.RepchainDataFrameHolder`
        Container holding all the results of a simulation without cutoff and all its parameters.
    max_duration : int
        Maximum duration entanglement is stored in memory.
    use_default_method : bool
        Determines whether default or nondefault implementation of the function should be used.
        The nondefault implementation may be faster under some circumstances.

    Returns
    -------
    :object:`~.repchain_dataframe_holder.RepchainDataFrameHolder`
        Container holding all the results of the simulation as if it were performed with a cutoff time.

    Note
    ----
    Currently only works for a single sequential quantum repeater.
    The input repchain_dataframe_holder must have "duration_between_alice_and_bob" as a varied parameter.
    There is both a default and nondefault implementation of this function. While they should have the same result,
    one may be more efficient than the other in some circumstances. Performance of both implementations has not
    yet been studied.

    Example
    -------

    >>> data = {"outcome_A": [0, 1, 0], "basis_A": ["Z", "Z", "X"],
    >>>         "outcome_B": [0, 1, 0], "basis_B": ["Z", "Z", "X"],
    >>>         "generation_duration": [100, 100, 100],
    >>>         "midpoint_outcome_0": [0, 0, 0], "midpoint_outcome_1": [0, 0, 0], "swap_outcome_0": [0, 0, 0],
    >>>         "duration_between_alice_and_bob": [10, 30, 15]}
    >>>
    >>> repchain_dataframe_holder = RepchainDataFrameHolder(data, generation_duration_unit="rounds")
    >>> repchain_dataframe_holder_with_cutoff = implement_cutoff_duration(repchain_dataframe_holder, 20)
    >>>
    >>> print(repchain_dataframe_holder_with_cutoff)
    >>> print(repchain_dataframe_holder_with_cutoff.baseline_parameters)
    >>>

     output:

        outcome_A  basis_A  outcome_B basis_B  generation_duration  midpoint_outcome_0  midpoint_outcome_1
     0          0        Z          0       Z                100.0                   0                   0
     2          0        X          0       X                190.0                   0                   0
        swap_outcome_0  duration_between_alice_and_bob
     0               0                            10
     2               0                            15

    {'cutoff_round': 20, 'generation_duration_unit': 'rounds'}

    If there were a cutoff of 20 rounds, during what is the second entanglement distribution (row)
    in the original data, entanglement would have been discarded 10 rounds before entanglement was successfully swapped.
    At this point, the setup would have to start all over again.
    To represent this in the data when implementing a cutoff time, it is assumed that at this point we are at the
    beginning of what in in the original data is the third entanglement distribution.
    In the data with cutoff time, there are now only two entanglement distributions,
    the second of which is basically the third of the orginal data,
    but it also includes rounds that were "wasted" by discarding entanglement.

    """

    def _implement_cutoff_duration_default(repchain_dataframe_holder, max_duration):
        """Function which changes a repchain_dataframe_holder such that it is as if there was a cutoff time.

        Parameters
        ----------
        repchain_dataframe_holder: ~.repchain_dataframe_holder.RepchainDataFrameHolder
            Container holding all the results of a simulation without cutoff and all its parameters.
        max_duration: int
            Maximum duration entanglement is stored in memory.

        Returns
        -------
        :object:`~.repchain_dataframe_holder.RepchainDataFrameHolder`
            Container holding all the results of the simulation as if it were performed with a cutoff time.

        Note
        ----
        Currently only works for a single sequential quantum repeater.
        The input repchain_dataframe_holder must have "duration_between_alice_and_bob" as a varied parameter.
        Default function used to implement :func:`~.repchain_data_cutoff.implement_cutoff_duration`

        """
        df = deepcopy(repchain_dataframe_holder.dataframe)

        # remove results to make sure last one does not exceed cutoff time (helps in next part)
        to_be_removed = []
        for index in reversed(df.index.values.tolist()):
            if df.loc[index, "duration_between_alice_and_bob"] > max_duration:
                to_be_removed.append(index)
            else:
                break
        df = df.drop(to_be_removed)

        # determine which results exceed cutoff and see how much "extra" duration this means for next result
        exceeding = df["duration_between_alice_and_bob"] > max_duration
        df.loc[exceeding, "carry_over"] = \
            df["generation_duration"] - df["duration_between_alice_and_bob"] + max_duration

        # add "extra" duration (carry over) to the next result
        carry_over = 0
        for row in zip(df[exceeding].index, df.loc[exceeding, "carry_over"]):
            next_index = row[0] + 1
            carry_over += row[1]
            if df.loc[next_index, "carry_over"] != df.loc[next_index, "carry_over"]:
                df.loc[next_index, "generation_duration"] += carry_over
                carry_over = 0

        # filter
        df = df[df["duration_between_alice_and_bob"] <= max_duration]
        df = df.drop("carry_over", 1)

        # rebuild repchain_dataframe_holder and add "cutoff_round" as baseline parameter
        new_baseline_parameters = deepcopy(repchain_dataframe_holder.baseline_parameters)
        new_baseline_parameters.update({"cutoff_round": max_duration})
        old_description = deepcopy(repchain_dataframe_holder.description)
        if old_description is None:
            old_description = ""
        new_description = old_description + "\n \n implemented cutoff in postprocessing"

        result = RepchainDataFrameHolder(number_of_nodes=repchain_dataframe_holder.number_of_nodes,
                                         baseline_parameters=new_baseline_parameters,
                                         description=new_description,
                                         data=df
                                         )

        return result

    def _implement_cutoff_duration_nondefault(repchain_dataframe_holder, max_duration):
        """Function which changes a repchain_dataframe_holder such that it is as if there were cutoff time.

        Parameters
        ----------
        repchain_dataframe_holder: :object:`~.repchain_dataframe_holder.RepchainDataFrameHolder`
            Container holding all the results of a simulation without cutoff and all its parameters.
        max_duration: int
            Maximum duration entanglement is stored in memory.

        Returns
        -------
        :object:`~.repchain_dataframe_holder.RepchainDataFrameHolder`
            Container holding all the results of the simulation as if it were performed with a cutoff time.

        Note
        ----
        Currently only works for a single sequential quantum repeater.
        The input repchain_dataframe_holder must have "duration_between_alice_and_bob" as a varied parameter.
        Nondefault function used to implement :func:`~.repchain_data_cutoff.implement_cutoff_duration`

        """

        df = deepcopy(repchain_dataframe_holder.dataframe)

        # first, we identify which rows of the dataframe contain a result that would have hit the cut-off time.

        # filtered dataframe
        filtered_df = df[df["duration_between_alice_and_bob"] > max_duration]

        # secondly, starting from the last result, we add the amount of cutoff duration
        # to the generation duration to the result after a result that hit the cutoff,
        # and remove the result that hit the cutoff.
        # This procedure ensures that while results that were cut off are filtered out,
        # the counting of the number of attempts remains correct.
        # Note that it is important to start from the bottom because consecutive results could be cut off.

        generation_duration_integer_position = df.columns.get_loc("generation_duration")
        for index in reversed(filtered_df.index.values.tolist()):
            row_integer_position = df.index.get_loc(index)
            row = df.iloc[row_integer_position]
            if row_integer_position + 1 != df.shape[0]:
                duration_before_state_in_memory = row["generation_duration"] - row["duration_between_alice_and_bob"]
                df.iat[row_integer_position + 1, generation_duration_integer_position] += \
                    duration_before_state_in_memory + max_duration

            df = df.drop(index)

        new_baseline_parameters = deepcopy(repchain_dataframe_holder.baseline_parameters)
        new_baseline_parameters.update({"cutoff_round": max_duration})
        old_description = deepcopy(repchain_dataframe_holder.description)
        if old_description is None:
            old_description = ""
        new_description = old_description + "\n \n implemented cutoff in postprocessing"

        result = RepchainDataFrameHolder(number_of_nodes=repchain_dataframe_holder.number_of_nodes,
                                         baseline_parameters=new_baseline_parameters,
                                         description=new_description,
                                         data=df
                                         )

        return result

    if use_default_method:
        warnings.warn('Choosing default of two methods for updating RepchainDataframeHolder with cut-off criterion. '
                      'The nondefault method might be faster.')
        return _implement_cutoff_duration_default(repchain_dataframe_holder=repchain_dataframe_holder,
                                                  max_duration=max_duration)
    else:
        warnings.warn('Choosing nondefault of two methods for updating RepchainDataframeHolder with cut-off criterion. '
                      'The default method might be faster.')
        return _implement_cutoff_duration_nondefault(repchain_dataframe_holder=repchain_dataframe_holder,
                                                     max_duration=max_duration)


def scan_cutoff(cutoff_duration_min, cutoff_duration_max, stepsize, use_default_method=True,
                target_dir="cutoff", filename="combined_state_data.pickle"):
    """Change simulation results as if a cutoff time had been used, for a range of different cutoff times.

    For a number of different values of the cutoff_duration, a repchain_dataframe_holder is modified such that
    it is as if the experiment would have been performed using a cutoff time corresponding to cutoff_duration
    duration of entanglement generation. Each resulting repchain_dataframe_holder is saved individually.

    Parameters
    ----------
    cutoff_duration_min: int
        lower bound on cutoffs to implement
    cutoff_duration_max: int
        upper bound on cutoffs to implement
    stepsize: int
        difference between different cutoffs to implement
    use_default_method : bool
        determines whether default or nondefault implementation of
        :func:`~.repchain_data_cutoff.implement_cutoff_duration` should be used.
        The nondefault implementation may be faster under some circumstances.
    target_dir: str (optional)
        name of directory to store results
    filename: str (optional)
        name of file holding pickled :object:`~.repchain_dataframe_holder.RepchainDataFrameHolder`
        to implement cutoff on

    Note
    ----
    Output :object:`~.repchain_dataframe_holder.RepchainDataFrameHolder` objects are pickled and saved in the target_dir
    with filenames "cutoff=*", where * = cutoff duration.
    Uses :func:`~.repchain_data_cutoff.implement_cutoff_duration` function
    (but is more efficient than just calling the function for each cutoff time individually,
    because results obtained with a larger cutoff time are reused when calculating results for a shorter cutoff time).
    Currently only works for a single sequential quantum repeater.
    The input repchain_dataframe_holder must have "duration_between_alice_and_bob" as a varied parameter.

    """
    start_time = time.time()

    if not os.path.isfile(filename):
        raise FileNotFoundError("File not found: {}".format(filename))
    repchain_dataframe_holder = pickle.load(open(filename, "rb"))

    # directory to save cutoff results
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    number_of_cutoffs = 0

    data = implement_cutoff_duration(repchain_dataframe_holder, cutoff_duration_max, use_default_method=use_default_method)
    pickle.dump(data, open("{}/cutoff={}.pickle".format(target_dir, cutoff_duration_max), "wb"))

    for cutoff in reversed(range(cutoff_duration_min, cutoff_duration_max, stepsize)):

        data = implement_cutoff_duration(data, cutoff, use_default_method=use_default_method)
        pickle.dump(data, open("{}/cutoff={}.pickle".format(target_dir, cutoff), "wb"))
        number_of_cutoffs += 1

    total_time = time.time() - start_time

    print("\n\nImplemented {} different cutoff times on {} results in {} seconds.\n"
          .format(number_of_cutoffs, repchain_dataframe_holder.number_of_results, total_time))


def process_bb84_cutoff(directory="cutoff"):
    """Combine results of scan_cutoff and perform BB84 processing (calculating e.g. SKR as function of cutoff)

    Parameters
    ----------
    directory: str (optional)
        name of directory containing pickled :object:`~.repchain_dataframe_holder.RepchainDataFrameHolder`s
        with different cutoffs.

    Note
    ----
    Stores pickled combined :object:`~.repchain_dataframe_holder.RepchainDataFrameHolder` as
    "combined_cutoff_data.pickle".
    The output data of QKD processing is saved as "cutoff_processed.csv".

    """
    if not os.path.exists(directory):
        raise NotADirectoryError("Directory not found: {}".format(directory))

    combine_data(raw_data_dir=directory, suffix=".pickle", output="combined_cutoff_data.pickle")

    combined_data = pickle.load(open("combined_cutoff_data.pickle", "rb"))
    results = process_repchain_dataframe_holder(repchain_dataframe_holder=combined_data,
                                                processing_functions=[process_data_bb84])

    results.to_csv("cutoff_processed.csv", index=False)


def show_cutoff_histo(filename="combined_state_data.pickle"):
    """Show a histogram of how often each amount of waiting time occurs in a repchain_dataframe_holder.

    Parameters
    ----------
    filename: str
        name of file holding pickled :object:`~.repchain_dataframe_holder.RepchainDataFrameHolder`

    Note
    ----
    Only works for single sequential quantum repeater.
    The input repchain_dataframe_holder must have "duration_between_alice_and_bob" as a varied parameter.

    """
    data = pickle.load(open(filename, "rb"))
    data.dataframe.hist(column="duration_between_alice_and_bob", bins=100)
    plt.show()
